from __future__ import annotations

import random

from .dsl import Case, ColumnSpec, Program, TableData


def generate_workflow_case(seed: int) -> Case:
    workflows = [
        _etl_cleanup_workflow,
        _log_aggregation_workflow,
        _feature_engineering_workflow,
        _join_enrichment_workflow,
        _null_heavy_workflow,
    ]
    builder = workflows[seed % len(workflows)]
    return builder(seed)


def _etl_cleanup_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 101)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["retail", "enterprise", "trial", ""]),
            "x": rnd.choice([0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 1.0, 3.5, 10.0, None]),
            "flag": rnd.choice([True, False, None]),
        }
        for idx in range(12)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-etl",
        seed,
        [
            {"op": "filter", "column": "flag", "cmp": "==", "value": True},
            {"op": "mutate", "column": "y_float", "expr": {"kind": "cast", "source": "y", "to": "float"}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "y_float", "func": "sum", "as": "sum_y"},
                    {"column": "x", "func": "count", "as": "count_x"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-etl", seed, [table], program)


def _log_aggregation_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 211)
    services = ["api", "worker", "frontend", "scheduler", None]
    rows = [
        {
            "id": idx,
            "g": rnd.choice(services),
            "x": rnd.choice([0, 1, 2, 5, 10, -1]),
            "s": rnd.choice(["INFO", "WARN", "ERROR", "error", ""]),
        }
        for idx in range(18)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-log",
        seed,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "level", "expr": {"kind": "string_lower", "source": "s"}},
            {
                "op": "groupby",
                "keys": ["g", "level"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "events"},
                    {"column": "x", "func": "sum", "as": "total_x"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-log", seed, [table], program)


def _feature_engineering_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 307)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["A", "B", "C", None]),
            "x": rnd.choice([-2, -1, 0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 0.5, 1.0, 2.5, None]),
            "s": rnd.choice(["alpha", "beta", "space value", "", None]),
        }
        for idx in range(14)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-feature",
        seed,
        [
            {"op": "mutate", "column": "x_shift", "expr": {"kind": "add_const", "source": "x", "value": 2}},
            {"op": "mutate", "column": "s_len", "expr": {"kind": "string_length", "source": "s"}},
            {"op": "filter", "column": "x_shift", "cmp": ">=", "value": 0},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x_shift", "func": "max", "as": "max_x_shift"},
                    {"column": "s_len", "func": "sum", "as": "sum_s_len"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-feature", seed, [table], program)


def _join_enrichment_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 401)
    fact_rows = [
        {
            "id": rnd.choice([idx, idx % 4, 0, 1]),
            "g": rnd.choice(["north", "south", "west", None]),
            "x": rnd.choice([0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 1.0, 4.5, None]),
        }
        for idx in range(16)
    ]
    dim_rows = [
        {
            "id": idx,
            "j": rnd.choice([0, 1, 5, 10, None]),
            "z": rnd.choice([0.0, 1.0, 2.0, 10.0, None]),
            "tag": rnd.choice(["gold", "silver", "bronze", None]),
        }
        for idx in range(6)
    ]
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
        ],
        fact_rows,
    )
    dim = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        dim_rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-join",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "tag_norm", "expr": {"kind": "string_lower", "source": "tag"}},
            {
                "op": "groupby",
                "keys": ["tag_norm"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "z", "func": "max", "as": "max_z"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-join", seed, [fact, dim], program)


def _null_heavy_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 503)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["known", "unknown", None]),
            "x": rnd.choice([None, None, 0, 1, 2, 10]),
            "y": rnd.choice([None, None, 0.0, 1.0, -1.0]),
            "flag": rnd.choice([True, False, None]),
        }
        for idx in range(20)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-null",
        seed,
        [
            {"op": "filter", "column": "g", "cmp": "!=", "value": "unknown"},
            {"op": "mutate", "column": "x_plus", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {
                "op": "groupby",
                "keys": ["flag"],
                "aggs": [
                    {"column": "x_plus", "func": "sum", "as": "sum_x_plus"},
                    {"column": "y", "func": "count", "as": "count_y"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-null", seed, [table], program)
