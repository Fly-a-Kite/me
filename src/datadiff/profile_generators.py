from __future__ import annotations

import random
from typing import Any

from .csv_roundtrip import DEFAULT_LONG_NUMERIC_CSV_VALUES, csv_long_numeric_values
from .dsl import Case, ColumnSpec, Program, TableData
from .identifiers import make_safe_output_name


def generate_null_groupby_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 32452843 + 19)
    string_values = [None, "", "a", "alpha", "space value", "delta"]
    row_count = rnd.randint(1, 8)
    rows = []
    for idx in range(row_count):
        value = string_values[idx % len(string_values)]
        if idx == 0:
            value = None
        rows.append(
            {
                "x": rnd.choice([-10, -1, 0, 1, 2, 10, None]),
                "s": value,
            }
        )
    table = TableData(
        "t0",
        [
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    ascending = rnd.choice([True, False])
    limit = rnd.randint(1, max(1, row_count + 2))
    program = Program(
        f"prog-{seed:08d}-null-groupby-topk",
        seed,
        [
            {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "s"}},
            {
                "op": "groupby",
                "keys": ["m_0"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["m_0"]},
            {"op": "sort", "columns": ["m_0"], "ascending": ascending},
            {"op": "limit", "n": limit},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-groupby-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_storage_offset_case(seed: int) -> Case:
    row_count = 300_000
    offset = 0 if seed % 2 == 0 else 200_000
    table = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False)],
        [{"id": idx} for idx in range(row_count)],
    )
    program = Program(
        f"prog-{seed:08d}-storage-offset",
        seed,
        [
            {"op": "sort", "columns": ["id"], "ascending": True},
            {"op": "offset", "n": offset},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-storage-offset",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "storage_offset",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22656",
            "row_count": row_count,
            "offset": offset,
        },
    )


def generate_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 67867967 + 23)
    row_count = rnd.randint(1, 8)
    rows = []
    group_values = ["b", "c", "d", "e"]
    for idx in range(row_count):
        if idx == 0:
            rows.append({"g": "a", "x": None})
            continue
        group = group_values[idx % len(group_values)]
        value = rnd.choice([None, -10, -1, 0, 1, 2, 5, 10])
        rows.append({"g": group, "x": value})
    table = TableData(
        "t0",
        [
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_x"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-null-agg-topk",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": row_count + 2},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-agg-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_filter_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 701408733 + 29)
    rows = [
        {"id": 0, "g": "a", "x": None, "lane": "keep"},
        {"id": 1, "g": "a", "x": None, "lane": "keep"},
        {"id": 2, "g": "b", "x": rnd.choice([1, 2, 5]), "lane": "keep"},
        {"id": 3, "g": "b", "x": rnd.choice([None, 0, 3]), "lane": "keep"},
        {"id": 4, "g": "c", "x": rnd.choice([-1, 0, 7]), "lane": "keep"},
        {"id": 5, "g": "drop", "x": rnd.choice([None, 9]), "lane": "skip"},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("lane", "str", nullable=False),
        ],
        rows,
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_x"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-filter-null-agg-topk",
        seed,
        [
            {"op": "filter", "column": "lane", "cmp": "!=", "value": "skip"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
            {"op": "select", "columns": ["g", "m_0"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "m_0", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-filter-null-agg-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_join_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 74649677 + 27)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1},
            {"id": 1, "g": "a", "x": None},
            {"id": 2, "g": "b", "x": 2},
            {"id": 3, "g": "c", "x": -1},
            {"id": 4, "g": "d", "x": 0},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 10, "tag": "alpha"},
            {"id": 0, "j": None, "tag": "beta"},
            {"id": 2, "j": 5, "tag": "gamma"},
            {"id": 2, "j": 7, "tag": "delta"},
            {"id": 5, "j": 9, "tag": "orphan"},
        ],
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_j"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-join-null-agg-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "j", "to": "float"}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "m_0", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": 8},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-agg-topk",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_join_null_key_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 82_589_933 + 43)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 1, "g": "a", "x": -1},
            {"id": 2, "g": "b", "x": 2},
            {"id": 3, "g": "c", "x": rnd.choice([0, 3, 5])},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 2, "j": rnd.choice([5, 7, 9]), "tag": "match"},
            {"id": 9, "j": rnd.choice([11, None]), "tag": "orphan"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-join-null-key-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "groupby",
                "keys": ["j"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["j"]},
            {
                "op": "sort",
                "keys": [
                    {
                        "column": "j",
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-key-topk",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "join_null_key_topk",
            "source_issue": "https://github.com/apache/datafusion/issues/22190",
        },
    )


def generate_wide_offset_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 99_991 + 47)
    payload_columns = [
        ColumnSpec(f"p_{idx}", "float" if idx % 3 == 0 else "int", nullable=True)
        for idx in range(18)
    ]
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("sort_key", "int", nullable=True),
        *payload_columns,
    ]
    row_count = 384
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row: dict[str, Any] = {
            "id": idx,
            "sort_key": None if idx % 97 == 0 else (row_count - idx + (idx % 7)),
        }
        for column in payload_columns:
            payload_idx = int(column.name.split("_", 1)[1])
            if idx % (payload_idx + 11) == 0:
                row[column.name] = None
            elif column.type == "float":
                row[column.name] = round((idx * (payload_idx + 1)) / 13.0, 6)
            else:
                row[column.name] = idx * (payload_idx + 1)
        rows.append(row)
    offset_n = rnd.randint(240, 340)
    limit_n = rnd.randint(1, 4)
    table = TableData("t0", columns, rows)
    program = Program(
        f"prog-{seed:08d}-wide-offset-topk",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": "sort_key", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": offset_n},
            {"op": "limit", "n": limit_n},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-wide-offset-topk",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "wide_offset_topk",
            "source_issue": "https://github.com/duckdb/duckdb/issues/11261",
            "row_count": row_count,
            "offset": offset_n,
            "limit": limit_n,
        },
    )


def generate_empty_filter_groupby_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_299_709 + 53)
    rows = [
        {"id": 0, "g": "a", "x": 1, "lane": "keep"},
        {"id": 1, "g": "a", "x": None, "lane": "keep"},
        {"id": 2, "g": "b", "x": 2, "lane": "skip"},
        {"id": 3, "g": None, "x": rnd.choice([3, None]), "lane": "skip"},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("lane", "str", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-empty-filter-groupby",
        seed,
        [
            {"op": "filter", "column": "lane", "cmp": "==", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["g", "count_x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_x", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-empty-filter-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "empty_filter_groupby",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/43767",
        },
    )


def generate_join_filter_groupby_case(seed: int) -> Case:
    rnd = random.Random(seed * 91815541 + 33)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1, "y": 0.5},
            {"id": 1, "g": "a", "x": 0, "y": -0.5},
            {"id": 1, "g": "b", "x": 2, "y": 1.0},
            {"id": 2, "g": "b", "x": -1, "y": 0.0},
            {"id": 3, "g": "c", "x": 3, "y": None},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 5, "z": 0.5, "tag": "alpha"},
            {"id": 1, "j": 7, "z": 1.0, "tag": "beta"},
            {"id": 1, "j": -2, "z": -0.5, "tag": "gamma"},
            {"id": 3, "j": 1, "z": 2.0, "tag": "delta"},
        ],
    )
    ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-filter-groupby",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "source": "m_0", "op": "mul", "value": 2}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "m_1", "func": "sum", "as": "sum_m_1"},
                    {"column": "j", "func": "count", "as": "count_j"},
                    {"column": "z", "func": "max", "as": "max_z"},
                ],
            },
            {"op": "select", "columns": ["g", "sum_m_1", "count_j", "max_z"]},
            {"op": "sort", "columns": ["sum_m_1", "count_j", "g"], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-filter-groupby",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_join_null_truth_filter_case(seed: int) -> Case:
    rnd = random.Random(seed * 97_409 + 37)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 1, "g": "a", "x": 10},
            {"id": 2, "g": "b", "x": 20},
            {"id": 3, "g": "c", "x": 30},
            {"id": 4, "g": "d", "x": 40},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 1, "j": 100, "z": 1.0, "tag": "p"},
            {"id": 2, "j": 200, "z": 2.0, "tag": "q"},
            {"id": 5, "j": 300, "z": 3.0, "tag": "r"},
        ],
    )
    threshold = rnd.choice([125, 150, 175])
    program = Program(
        f"prog-{seed:08d}-join-null-truth-filter",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "j", "cmp": "gt_is_not_true", "value": threshold},
            {"op": "select", "columns": ["id", "g", "j", "tag"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-truth-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "join_null_truth_filter",
            "source_issue": "https://github.com/apache/datafusion/issues/22441",
        },
    )


def generate_join_groupby_stress_case(seed: int) -> Case:
    row_count = 100_000
    edges = _stable_graph_edges(row_count)
    left_rows = [{"fromnode": source, "tonode": target} for source, target in edges]
    path_rows = [{"p6_fromnode": source, "p6_tonode": target} for source, target in edges]
    probe_rows = [
        {
            "p4_fromnode": source,
            "p4_tonode_join": target,
            "p4_tonode": target,
        }
        for source, target in edges
    ]
    left = TableData(
        "t0",
        [
            ColumnSpec("fromnode", "int", nullable=False),
            ColumnSpec("tonode", "int", nullable=False),
        ],
        left_rows,
    )
    path = TableData(
        "t1",
        [
            ColumnSpec("p6_fromnode", "int", nullable=False),
            ColumnSpec("p6_tonode", "int", nullable=False),
        ],
        path_rows,
    )
    probe = TableData(
        "t2",
        [
            ColumnSpec("p4_fromnode", "int", nullable=False),
            ColumnSpec("p4_tonode_join", "int", nullable=False),
            ColumnSpec("p4_tonode", "int", nullable=False),
        ],
        probe_rows,
    )
    program = Program(
        f"prog-{seed:08d}-join-groupby-stress",
        seed,
        [
            {
                "op": "join",
                "table": "t1",
                "left_on": "tonode",
                "right_on": "p6_fromnode",
                "how": "inner",
            },
            {
                "op": "groupby",
                "keys": ["fromnode", "tonode"],
                "aggs": [{"column": "p6_tonode", "func": "count", "as": "path2_count"}],
            },
            {
                "op": "join",
                "table": "t2",
                "left_on": "fromnode",
                "right_on": "p4_tonode_join",
                "how": "inner",
            },
            {
                "op": "groupby",
                "keys": ["p4_fromnode", "p4_tonode"],
                "aggs": [{"column": "path2_count", "func": "sum", "as": "path3_count"}],
            },
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "path3_count", "func": "sum", "as": "total_path3_count"},
                    {"column": "path3_count", "func": "count", "as": "path3_group_count"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-groupby-stress",
        seed=seed,
        tables=[left, path, probe],
        program=program,
        metadata={"generator_profile": "join_groupby_stress", "row_count": row_count},
    )


def _stable_graph_edges(row_count: int) -> list[tuple[int, int]]:
    left_span = row_count // 2 + row_count // 4
    right_span = row_count // 2 + row_count
    edges: list[tuple[int, int]] = []
    for idx in range(row_count):
        left_seed = idx * 1_000_003
        right_seed = idx * 9_176
        if _stable_hash64(left_seed) == _stable_hash64(right_seed):
            continue
        source = min(_stable_hash64(left_seed) % left_span, _stable_hash64(left_seed + 7) % left_span)
        target = min(_stable_hash64(right_seed) % right_span, _stable_hash64(right_seed + 11) % right_span)
        edges.append((int(source), int(target)))
    return edges


def _stable_hash64(value: int) -> int:
    mask = (1 << 64) - 1
    value &= mask
    value ^= value >> 32
    value = (value * 0xD6E8FEB86659FD93) & mask
    value ^= value >> 32
    value = (value * 0xD6E8FEB86659FD93) & mask
    value ^= value >> 32
    return value & mask


def generate_float_group_key_case(seed: int) -> Case:
    rnd = random.Random(seed * 86028121 + 31)
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 0, "g": None, "x": None, "y": 0.5, "flag": True, "s": "scnvvnMt"},
            {"id": 0, "g": "", "x": 0, "y": -0.5, "flag": None, "s": None},
            {"id": 0, "g": "cn", "x": 0, "y": 0.0, "flag": None, "s": "a"},
            {"id": 0, "g": "A", "x": -18, "y": 0.0, "flag": True, "s": "space value"},
        ],
    )
    join_table = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": -10, "z": -0.5, "tag": "space value"},
            {"id": 1, "j": None, "z": 0.5, "tag": "A"},
            {"id": 0, "j": 0, "z": 1.0, "tag": "beta"},
            {"id": 0, "j": 2, "z": -0.5, "tag": "beta"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-float-group-key",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
            {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
            {
                "op": "sort",
                "columns": ["s", "flag", "g", "id", "j", "m_0", "tag", "x", "y", "z"],
                "ascending": rnd.choice([True, False]),
            },
            {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "source": "m_0", "op": "mul", "value": 10}},
            {"op": "mutate", "column": "m_2", "expr": {"kind": "cast", "source": "m_0", "to": "float"}},
            {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "source": "m_1", "op": "div", "value": 3}},
            {
                "op": "groupby",
                "keys": ["m_3"],
                "aggs": [{"column": "m_2", "func": "min", "as": "min_m_2"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-float-group-key",
        seed=seed,
        tables=[table, join_table],
        program=program,
    )


def generate_join_null_sort_case(seed: int) -> Case:
    rnd = random.Random(seed * 9999991 + 41)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 0, "x": 2, "y": 0.5, "s": "alpha"},
            {"id": 1, "x": -1, "y": None, "s": "Beta"},
            {"id": 1, "x": 0, "y": -0.5, "s": "space value"},
            {"id": 2, "x": 3, "y": 1.0, "s": "gamma"},
            {"id": 4, "x": None, "y": 0.0, "s": ""},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 1, "j": None, "z": 1.0, "tag": "Alpha"},
            {"id": 1, "j": 2, "z": None, "tag": "beta"},
            {"id": 2, "j": -2, "z": -0.5, "tag": "MIXED"},
            {"id": 3, "j": 7, "z": 0.5, "tag": "orphan"},
        ],
    )
    ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-null-sort",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "j", "to": "float"}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "string_length", "source": "tag"}},
            {"op": "select", "columns": ["id", "x", "y", "s", "tag", "m_0", "m_1"]},
            {"op": "sort", "columns": ["m_0", "m_1", "id", "s"], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-sort",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_ordered_groupby_sort_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_000_003 + 59)
    groups = ["a", "b", "c", "d"]
    rows: list[dict[str, Any]] = []
    for idx in range(8):
        group = groups[idx % len(groups)]
        if idx == 0:
            group = "a"
        value_choices = [None, -2, -1, 0, 1, 2, 5, 10]
        rows.append(
            {
                "id": idx,
                "g": group,
                "x": rnd.choice(value_choices),
                "z": rnd.choice([None, -1, 0, 1, 2, 4, 8]),
            }
        )
    # Ensure two non-null aggregate outputs whose input-order and value-order
    # disagree. This exercises stale sortedness/ordering metadata generally,
    # without hard-coding a backend-specific oracle.
    rows[0] = {"id": 0, "g": "a", "x": 1, "z": 1}
    rows[1] = {"id": 1, "g": "b", "x": 2, "z": 2}
    rows[2] = {"id": 2, "g": "b", "x": 0, "z": 0}
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("z", "int", nullable=True),
        ],
        rows,
    )
    value_col = rnd.choice(["x", "z"])
    agg_func = rnd.choice(["max", "min"])
    alias = f"{agg_func}_{value_col}"
    pre_sort_ascending = rnd.choice([True, False])
    post_sort_ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-ordered-groupby-sort",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": value_col, "ascending": pre_sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": value_col, "func": agg_func, "as": alias}],
            },
            {"op": "select", "columns": ["g", alias]},
            {
                "op": "sort",
                "keys": [
                    {"column": alias, "ascending": post_sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-ordered-groupby-sort",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "ordered_groupby_sort"},
    )


def generate_topk_resort_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_618_033 + 61)
    rows = []
    for idx in range(14):
        rows.append(
            {
                "id": idx,
                "g": rnd.choice(["a", "b", "c", None]),
                "x": rnd.choice([None, -10, -2, -1, 0, 1, 2, 10]),
                "z": rnd.choice([None, -3, 0, 1, 3, 8]),
                "s": rnd.choice(["", "A", "a", "space value", None]),
            }
        )
    rows[0] = {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"}
    rows[1] = {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"}
    rows[2] = {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""}
    table = TableData(
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
    first_col = rnd.choice(["x", "z", "g", "s"])
    second_col = rnd.choice([col for col in ["x", "z", "g", "s"] if col != first_col])
    limit_n = rnd.randint(1, 6)
    offset_n = rnd.randint(0, 2)
    program = Program(
        f"prog-{seed:08d}-topk-resort",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": first_col, "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": limit_n},
            {
                "op": "sort",
                "keys": [
                    {"column": second_col, "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": offset_n},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-topk-resort",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "topk_resort", "limit": limit_n, "offset": offset_n},
    )


def generate_join_ordered_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 2_147_483 + 67)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1},
            {"id": 1, "g": "b", "x": 2},
            {"id": 1, "g": "b", "x": 0},
            {"id": 2, "g": "c", "x": None},
            {"id": 3, "g": "d", "x": -1},
            {"id": 4, "g": None, "x": 5},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 10, "z": 1, "tag": "alpha"},
            {"id": 1, "j": 2, "z": None, "tag": "beta"},
            {"id": 1, "j": 7, "z": 3, "tag": "Beta"},
            {"id": 2, "j": None, "z": 8, "tag": "space value"},
            {"id": 5, "j": 9, "z": -1, "tag": "orphan"},
        ],
    )
    agg_col = rnd.choice(["j", "z", "x"])
    agg_func = rnd.choice(["min", "max", "sum", "count"])
    alias = f"{agg_func}_{agg_col}"
    sort_ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-ordered-agg-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "sort",
                "keys": [
                    {"column": "j", "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": agg_col, "func": agg_func, "as": alias}],
            },
            {"op": "select", "columns": ["g", alias]},
            {
                "op": "sort",
                "keys": [
                    {"column": alias, "ascending": sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(1, 6)},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-ordered-agg-topk",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={"generator_profile": "join_ordered_agg_topk"},
    )


def generate_global_null_aggregate_case(seed: int) -> Case:
    empty_input = seed % 2 == 0
    rows = [] if empty_input else [
        {"id": 0, "g": "a", "x": None, "y": None},
        {"id": 1, "g": "b", "x": None, "y": None},
        {"id": 2, "g": None, "x": None, "y": 5},
        {"id": 3, "g": "space value", "x": None, "y": -1},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-global-null-aggregate",
        seed,
        [
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "x", "func": "min", "as": "min_x"},
                    {"column": "x", "func": "max", "as": "max_x"},
                    {"column": "x", "func": "count", "as": "count_x"},
                    {"column": "y", "func": "sum", "as": "sum_y"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": True, "nulls": "first"},
                    {"column": "count_x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 1},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-global-null-aggregate",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "global_null_aggregate", "empty_input": empty_input},
    )


def generate_string_count_groupby_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "A", "x": 1},
        {"id": 1, "g": "alpha", "s": None, "x": 2},
        {"id": 2, "g": "alpha", "s": "", "x": None},
        {"id": 3, "g": "beta", "s": "space value", "x": -1},
        {"id": 4, "g": "beta", "s": "中文", "x": 0},
        {"id": 5, "g": None, "s": None, "x": 5},
        {"id": 6, "g": None, "s": "delta", "x": None},
        {"id": 7, "g": "gamma", "s": None, "x": 3},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "s": "", "x": -3})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-string-count-groupby",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "count", "as": "count_s"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_s", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-count-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "string_count_groupby"},
    )


def generate_unique_count_groupby_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1},
        {"id": 1, "g": "alpha", "s": "red", "x": 1},
        {"id": 2, "g": "alpha", "s": None, "x": None},
        {"id": 3, "g": "beta", "s": "", "x": 2},
        {"id": 4, "g": "beta", "s": "space value", "x": 3},
        {"id": 5, "g": "beta", "s": "", "x": 2},
        {"id": 6, "g": None, "s": None, "x": None},
        {"id": 7, "g": None, "s": "中文", "x": 4},
        {"id": 8, "g": "gamma", "s": None, "x": None},
    ]
    if seed % 2:
        rows.append({"id": 9, "g": "gamma", "s": "red", "x": 4})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-unique-count-groupby",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "nunique", "as": "uniq_s_count"},
                    {"column": "x", "func": "nunique", "as": "uniq_x_count"},
                    {"column": "s", "func": "count", "as": "count_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "uniq_s_count", "ascending": False, "nulls": "last"},
                    {"column": "uniq_x_count", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-unique-count-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "unique_count_groupby",
            "source_issue": "https://github.com/apache/arrow/issues/36149",
        },
    )


def generate_bool_null_groupby_agg_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "flag": True, "x": 1},
        {"id": 1, "g": "alpha", "flag": None, "x": 2},
        {"id": 2, "g": "alpha", "flag": False, "x": None},
        {"id": 3, "g": "beta", "flag": None, "x": -1},
        {"id": 4, "g": "beta", "flag": None, "x": 0},
        {"id": 5, "g": None, "flag": True, "x": 5},
        {"id": 6, "g": None, "flag": False, "x": None},
        {"id": 7, "g": "gamma", "flag": None, "x": 3},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "flag": True, "x": -3})
    if seed % 3 == 0:
        rows.append({"id": 9, "g": "delta", "flag": False, "x": 4})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-bool-null-groupby-agg",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "flag", "func": "any", "as": "flag_any_value"},
                    {"column": "flag", "func": "all", "as": "flag_all_value"},
                    {"column": "flag", "func": "min", "as": "flag_min_value"},
                    {"column": "flag", "func": "max", "as": "flag_max_value"},
                    {"column": "flag", "func": "count", "as": "flag_seen_count"},
                    {"column": "flag", "func": "nunique", "as": "flag_distinct_count"},
                    {"column": "x", "func": "sum", "as": "x_sum_value"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_seen_count", "ascending": False, "nulls": "last"},
                    {"column": "flag_any_value", "ascending": False, "nulls": "last"},
                    {"column": "flag_all_value", "ascending": True, "nulls": "last"},
                    {"column": "flag_min_value", "ascending": True, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-null-groupby-agg",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bool_null_groupby_agg",
            "source_issue": "https://github.com/pola-rs/polars/issues/26671",
        },
    )


def generate_large_int_filter_groupby_case(seed: int) -> Case:
    high = 9_007_199_254_740_992
    rows = [
        {"id": 0, "g": "alpha", "x": high - 1, "flag": True},
        {"id": 1, "g": "alpha", "x": high, "flag": False},
        {"id": 2, "g": "alpha", "x": high + 1, "flag": None},
        {"id": 3, "g": "beta", "x": -high + 1, "flag": True},
        {"id": 4, "g": "beta", "x": -high, "flag": False},
        {"id": 5, "g": None, "x": 0, "flag": None},
        {"id": 6, "g": None, "x": 42, "flag": True},
        {"id": 7, "g": "gamma", "x": None, "flag": False},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "x": high + 3, "flag": True})
    if seed % 3 == 0:
        rows.append({"id": 9, "g": "delta", "x": -high - 3, "flag": None})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    filter_mode = seed % 3
    if filter_mode == 0:
        filter_op = {"op": "filter", "column": "x", "cmp": "in_set", "value": [high - 1, high + 1, -high, 0]}
    elif filter_mode == 1:
        filter_op = {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-high, high]}
    else:
        filter_op = {"op": "filter", "column": "x", "cmp": "!=", "value": 42}
    program = Program(
        f"prog-{seed:08d}-large-int-filter-groupby",
        seed,
        [
            filter_op,
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "count", "as": "x_seen_count"},
                    {"column": "x", "func": "min", "as": "x_min_value"},
                    {"column": "x", "func": "max", "as": "x_max_value"},
                    {"column": "flag", "func": "count", "as": "flag_seen_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "x_max_value", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-large-int-filter-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "large_int_filter_groupby",
            "source_issue": "https://github.com/pola-rs/polars/issues/27726",
            "source_issue_alt": "https://github.com/duckdb/duckdb/issues/22676",
        },
    )


def generate_set_membership_filter_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1, "flag": True},
        {"id": 1, "g": "alpha", "s": "", "x": 2, "flag": False},
        {"id": 2, "g": "alpha", "s": None, "x": None, "flag": None},
        {"id": 3, "g": "beta", "s": "blue", "x": 2, "flag": True},
        {"id": 4, "g": "beta", "s": "中文", "x": 3, "flag": False},
        {"id": 5, "g": "beta", "s": "space value", "x": None, "flag": True},
        {"id": 6, "g": None, "s": "", "x": 4, "flag": None},
        {"id": 7, "g": None, "s": "red", "x": 4, "flag": False},
        {"id": 8, "g": "gamma", "s": None, "x": 5, "flag": True},
    ]
    if seed % 2:
        rows.append({"id": 9, "g": "gamma", "s": "中文", "x": 5, "flag": False})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-set-membership-filter",
        seed,
        [
            {"op": "filter", "column": "s", "cmp": "in_set", "value": ["red", "", "中文"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "count", "as": "count_s"},
                    {"column": "x", "func": "nunique", "as": "uniq_x_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_s", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-set-membership-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "set_membership_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/22149",
        },
    )


def generate_pyarrow_groupby_filter_cast_membership_case(seed: int) -> Case:
    bucket_code = seed % 3
    other_bucket = (bucket_code + 1) % 3
    last_bucket = (bucket_code + 2) % 3
    membership_values = [float(bucket_code) + 0.5, float(-(bucket_code + 1)), 10.0]
    rows = [
        {"bucket_code": bucket_code, "amount_code": 10, "metric_value": -0.5},
        {"bucket_code": bucket_code, "amount_code": 2, "metric_value": -1.0},
        {"bucket_code": other_bucket, "amount_code": 10, "metric_value": 0.5},
        {"bucket_code": last_bucket, "amount_code": None, "metric_value": 1.0},
    ]
    if seed % 2:
        rows.append({"bucket_code": other_bucket, "amount_code": 2, "metric_value": None})
    table = TableData(
        "t0",
        [
            ColumnSpec("bucket_code", "int", nullable=False),
            ColumnSpec("amount_code", "int", nullable=True),
            ColumnSpec("metric_value", "float", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-pyarrow-groupby-filter-cast-membership",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["bucket_code", "amount_code"],
                "aggs": [
                    {"column": "bucket_code", "func": "min", "as": "agg_min_bucket"},
                    {"column": "amount_code", "func": "max", "as": "agg_max_amount"},
                ],
            },
            {"op": "filter", "column": "agg_min_bucket", "cmp": "in_set", "value": membership_values},
            {
                "op": "sort",
                "keys": [
                    {"column": "agg_min_bucket", "ascending": True, "nulls": "last"},
                    {"column": "amount_code", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["bucket_code", "amount_code", "agg_min_bucket", "agg_max_amount"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-groupby-filter-cast-membership",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_groupby_filter_cast_membership",
            "membership_values": membership_values,
        },
    )


def generate_null_predicate_filter_case(seed: int) -> Case:
    use_is_null = seed % 2 == 0
    predicate = "is_null" if use_is_null else "is_not_null"
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1, "flag": True},
        {"id": 1, "g": "alpha", "s": None, "x": 2, "flag": False},
        {"id": 2, "g": "beta", "s": "", "x": None, "flag": None},
        {"id": 3, "g": "beta", "s": "blue", "x": 2, "flag": True},
        {"id": 4, "g": None, "s": None, "x": 3, "flag": False},
        {"id": 5, "g": None, "s": "中文", "x": None, "flag": None},
        {"id": 6, "g": "gamma", "s": "space value", "x": 4, "flag": True},
        {"id": 7, "g": "gamma", "s": None, "x": 4, "flag": False},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-null-predicate-filter",
        seed,
        [
            {"op": "filter", "column": "s", "cmp": predicate, "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-predicate-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "null_predicate_filter",
            "predicate": predicate,
            "source_issue": "https://github.com/duckdb/duckdb/issues/4978",
        },
    )


def generate_boolean_predicate_filter_case(seed: int) -> Case:
    predicates = ["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]
    predicate = predicates[seed % len(predicates)]
    rows = [
        {"id": 0, "g": "alpha", "flag": True, "x": 1, "s": "red"},
        {"id": 1, "g": "alpha", "flag": False, "x": 2, "s": ""},
        {"id": 2, "g": "alpha", "flag": None, "x": None, "s": None},
        {"id": 3, "g": "beta", "flag": True, "x": 2, "s": "blue"},
        {"id": 4, "g": "beta", "flag": False, "x": 3, "s": "中文"},
        {"id": 5, "g": "beta", "flag": None, "x": None, "s": "space value"},
        {"id": 6, "g": None, "flag": True, "x": 4, "s": "red"},
        {"id": 7, "g": None, "flag": False, "x": 4, "s": None},
        {"id": 8, "g": "gamma", "flag": None, "x": 5, "s": ""},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-boolean-predicate-filter",
        seed,
        [
            {"op": "filter", "column": "flag", "cmp": predicate, "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-boolean-predicate-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "boolean_predicate_filter",
            "predicate": predicate,
            "source_issue": "https://github.com/pola-rs/polars/issues/8516",
            "source_issue_alt": "https://github.com/apache/datafusion/issues/22441",
        },
    )


def generate_post_topk_range_filter_case(seed: int) -> Case:
    limit_n = 6 if seed % 2 == 0 else 7
    rows = [
        {"id": 0, "g": "top", "score": 100, "x": -5, "flag": True},
        {"id": 1, "g": "top", "score": 95, "x": 7, "flag": False},
        {"id": 2, "g": "keep", "score": 90, "x": 1, "flag": None},
        {"id": 3, "g": "top", "score": 85, "x": None, "flag": True},
        {"id": 4, "g": "keep", "score": 80, "x": 2, "flag": False},
        {"id": 5, "g": "top", "score": 75, "x": -3, "flag": None},
        {"id": 6, "g": "late", "score": 70, "x": 0, "flag": True},
        {"id": 7, "g": "late", "score": 65, "x": 2, "flag": False},
        {"id": 8, "g": "late", "score": 60, "x": 1, "flag": None},
        {"id": 9, "g": None, "score": None, "x": 1, "flag": True},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-post-topk-range-filter",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": limit_n},
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 2]},
            {"op": "select", "columns": ["id", "g", "score", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-post-topk-range-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "post_topk_range_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/26803",
            "source_issue_alt": "https://github.com/duckdb/duckdb/issues/22075",
            "limit": limit_n,
        },
    )


def generate_tuple_absence_filter_case(seed: int) -> Case:
    left_rows = [
        {"row_id": 0, "a": 1, "b": 1, "payload": "matched"},
        {"row_id": 1, "a": 2, "b": 2, "payload": "survivor"},
        {"row_id": 2, "a": 3, "b": None, "payload": "unknown-left"},
        {"row_id": 3, "a": None, "b": 4, "payload": "unknown-both"},
    ]
    if seed % 2:
        left_rows.append({"row_id": 4, "a": 5, "b": 4, "payload": "definite-false"})
    left = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("a", "int", nullable=True),
            ColumnSpec("b", "int", nullable=True),
            ColumnSpec("payload", "str", nullable=True),
        ],
        left_rows,
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("a", "int", nullable=True),
            ColumnSpec("b", "int", nullable=True),
        ],
        [
            {"a": 1, "b": 1},
            {"a": None, "b": 4},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-tuple-absence-filter",
        seed,
        [
            {"op": "tuple_absence_filter", "columns": ["a", "b"], "table": "t1", "right_columns": ["a", "b"]},
            {"op": "select", "columns": ["row_id", "a", "b", "payload"]},
            {"op": "sort", "keys": [{"column": "row_id", "ascending": True, "nulls": "last"}]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-tuple-absence-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "tuple_absence_filter",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22418",
        },
    )


def generate_row_value_absence_filter_case(seed: int) -> Case:
    left_rows = [
        {"sample_id": 0, "left_a": 1, "left_b": 1, "label_value": "matched"},
        {"sample_id": 1, "left_a": 2, "left_b": 2, "label_value": "survivor"},
        {"sample_id": 2, "left_a": 3, "left_b": None, "label_value": "null-left"},
        {"sample_id": 3, "left_a": None, "left_b": 4, "label_value": "null-pair"},
    ]
    if seed % 2:
        left_rows.append({"sample_id": 4, "left_a": 5, "left_b": 4, "label_value": "unknown-tail"})
    left = TableData(
        "t0",
        [
            ColumnSpec("sample_id", "int", nullable=False),
            ColumnSpec("left_a", "int", nullable=True),
            ColumnSpec("left_b", "int", nullable=True),
            ColumnSpec("label_value", "str", nullable=True),
        ],
        left_rows,
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("right_a", "int", nullable=True),
            ColumnSpec("right_b", "int", nullable=True),
        ],
        [
            {"right_a": 1, "right_b": 1},
            {"right_a": None, "right_b": 4},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-row-value-absence-filter",
        seed,
        [
            {
                "op": "tuple_absence_filter",
                "columns": ["left_a", "left_b"],
                "table": "t1",
                "right_columns": ["right_a", "right_b"],
            },
            {"op": "select", "columns": ["sample_id", "left_a", "left_b", "label_value"]},
            {"op": "sort", "keys": [{"column": "sample_id", "ascending": True, "nulls": "last"}]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-row-value-absence-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={"generator_profile": "row_value_absence_filter"},
    )


def generate_running_sum_precision_case(seed: int) -> Case:
    row_count = 20_000
    increment = 0.0005
    rows = [{"row_id": idx, "x": increment} for idx in range(row_count)]
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("x", "float", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-running-sum-precision",
        seed,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
                "input_dtype": "float32",
            },
            {
                "op": "sort",
                "keys": [{"column": "row_id", "ascending": False, "nulls": "last"}],
            },
            {"op": "limit", "n": 1},
            {"op": "select", "columns": ["row_id", "run_x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-running-sum-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "running_sum_precision",
            "source_issue": "https://github.com/pola-rs/polars/issues/27662",
            "source_issue_alt": "https://github.com/pola-rs/polars/issues/26800",
            "row_count": row_count,
            "increment": increment,
            "expected_total": row_count * increment,
        },
    )


def generate_partitioned_running_sum_case(seed: int) -> Case:
    rnd = random.Random(seed * 4001 + 17)
    rows = []
    row_id = 0
    for group in ["A", "B", None]:
        for seq, value in enumerate([1.0, None, 2.5, -0.5], start=1):
            adjusted = value if value is None else value + (0.25 if group == "B" else 0.0)
            rows.append({"row_id": row_id, "grp": group, "seq": seq, "x": adjusted})
            row_id += 1
    rnd.shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("seq", "int", nullable=False),
            ColumnSpec("x", "float", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-partitioned-running-sum",
        seed,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "row_id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "float64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "row_id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["grp", "seq", "run_x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-partitioned-running-sum",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "partitioned_running_sum",
            "issue_inspiration": "partitioned ROWS-frame cumulative aggregates",
        },
    )


def generate_path_basename_keyed_pick_case(seed: int) -> Case:
    rnd = random.Random(seed * 5021 + 31)
    path_values = [
        "C:/warehouse/january/report.csv",
        "/tmp/releases/build.tar",
        "relative/name.parquet",
        "plain_name.txt",
        r"D:\archive\delta.log",
        None,
    ]
    rows = []
    row_id = 0
    for group in ["A", "B", None]:
        for offset, path_value in enumerate(path_values):
            rows.append(
                {
                    "row_id": row_id,
                    "grp": group,
                    "pick_key": (offset * 3 + row_id) % 11,
                    "path_value": path_value,
                    "payload": row_id % 5,
                }
            )
            row_id += 1
    rnd.shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("pick_key", "int", nullable=False),
            ColumnSpec("path_value", "str", nullable=True),
            ColumnSpec("payload", "int", nullable=False),
        ],
        rows,
    )
    basename_column = make_safe_output_name(
        "path_base_value",
        used={column.name for column in table.columns},
    )
    row_limit_cmp = "<=" if seed % 3 == 0 else "=="
    row_limit_value = 2 if row_limit_cmp == "<=" else 1
    pick_keys = [{"column": "pick_key", "ascending": True, "nulls": "last"}]
    output_sort_keys = [
        {"column": "pick_key", "ascending": True, "nulls": "last"},
        {"column": "row_id", "ascending": True, "nulls": "last"},
    ]
    program = Program(
        f"prog-{seed:08d}-path-basename-keyed-pick",
        seed,
        [
            {
                "op": "mutate",
                "column": basename_column,
                "expr": {"kind": "string_basename", "source": "path_value"},
            },
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": pick_keys,
                "cmp": row_limit_cmp,
                "value": row_limit_value,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    *output_sort_keys,
                ],
            },
            {"op": "select", "columns": ["grp", "pick_key", basename_column]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-path-basename-keyed-pick",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "path_basename_keyed_pick",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22849",
            "issue_inspiration": "path-string projection composed with keyed row selection",
        },
    )


def generate_sortedness_null_placement_case(seed: int) -> Case:
    rows = [
        {"row_id": 0, "x": 3},
        {"row_id": 1, "x": 1},
        {"row_id": 2, "x": 2},
        {"row_id": 3, "x": None},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    alias = make_safe_output_name("sorted_ok_x", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-sortedness-null-placement",
        seed,
        [
            {
                "op": "sort",
                "keys": [{"column": "x", "ascending": True, "nulls": "last"}],
            },
            {
                "op": "sortedness_check",
                "column": "x",
                "as": alias,
                "ascending": True,
                "nulls": "first",
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-sortedness-null-placement",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "sortedness_null_placement",
            "source_issue": "https://github.com/pola-rs/polars/issues/26993",
            "expected_sortedness": False,
        },
    )


def generate_simple_case_random_subject_case(seed: int) -> Case:
    row_count = 100_000
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("unexpected_else_seen", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-simple-case-random-subject",
        seed,
        [
            {
                "op": "random_case_probe",
                "as": alias,
                "rows": row_count,
                "branches": 3,
            }
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-simple-case-random-subject",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "simple_case_random_subject",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22576",
            "row_count": row_count,
            "expected_unexpected_else_seen": False,
        },
    )


def generate_group_quantile_key_probe_case(seed: int) -> Case:
    values = [1, 2, 3]
    quantiles = [0.0, 0.5, 1.0]
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("quantile_key_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-group-quantile-key-probe",
        seed,
        [
            {
                "op": "group_quantile_probe",
                "as": alias,
                "values": values,
                "quantiles": quantiles,
            }
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-group-quantile-key-probe",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "group_quantile_key_probe",
            "source_issue": "https://github.com/pola-rs/polars/issues/25888",
            "expected_quantiles": [1.0, 2.0, 3.0],
            "expected_quantile_key_mismatch": False,
        },
    )


def generate_scalar_subquery_double_parentheses_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("scalar_subquery_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-scalar-subquery-double-parentheses",
        seed,
        [{"op": "scalar_subquery_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-scalar-subquery-double-parentheses",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "scalar_subquery_double_parentheses",
            "source_issue": "https://github.com/duckdb/duckdb/issues/19851",
            "expected_scalar_subquery_mismatch": False,
            "expected_rows": [[2]],
        },
    )


def generate_window_avg_rows_frame_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("window_avg_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-window-avg-rows-frame",
        seed,
        [{"op": "window_avg_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-window-avg-rows-frame",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "window_avg_rows_frame",
            "source_issue": "https://github.com/pola-rs/polars/issues/26065",
            "expected_window_avg_mismatch": False,
            "expected_window_avg_values": [10.0, 15.0, 20.0, 25.0, 30.0],
        },
    )


def generate_struct_distinct_unnest_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("struct_distinct_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-struct-distinct-unnest",
        seed,
        [{"op": "struct_distinct_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-struct-distinct-unnest",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "struct_distinct_unnest",
            "source_issue": "https://github.com/duckdb/duckdb/issues/17278",
            "expected_struct_distinct_mismatch": False,
            "expected_rows": [["0", "0"], ["0", "1"]],
        },
    )


def generate_bit_compare_unequal_length_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("bit_compare_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-bit-compare-unequal-length",
        seed,
        [{"op": "bit_compare_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-bit-compare-unequal-length",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bit_compare_unequal_length",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22527",
            "expected_bit_compare_mismatch": False,
            "expected_bit_less": True,
        },
    )


def generate_round_even_float_scale_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("round_even_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-round-even-float-scale",
        seed,
        [{"op": "round_even_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-round-even-float-scale",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "round_even_float_scale",
            "source_issue": "https://github.com/duckdb/duckdb/issues/19491",
            "expected_round_even_mismatch": False,
            "expected_round_even_value": 2.67,
        },
    )


def generate_duckdb_float_literal_precision_case(seed: int) -> Case:
    literals = [
        "0.41000000000000003",
        "0.9999999999999999",
        "0.1000000000000000055511151231257827",
    ]
    literal = literals[seed % len(literals)]
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("float_literal_precision_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-float-literal-precision",
        seed,
        [{"op": "float_literal_precision_probe", "as": alias, "literal": literal}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-float-literal-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_float_literal_precision",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22837",
            "issue_inspiration": "decimal float literal and string cast should round identically",
            "literal": literal,
            "expected_float_literal_precision_mismatch": False,
        },
    )


def generate_polars_timestamp_precision_filter_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("timestamp_precision_filter_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-timestamp-precision-filter",
        seed,
        [{"op": "timestamp_precision_filter_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-timestamp-precision-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_timestamp_precision_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/27726",
            "issue_inspiration": "timestamp filter should compare us columns and ns literals without lossy downcast",
            "expected_timestamp_precision_filter_rows": 1,
        },
    )


def generate_series_rtruediv_operand_order_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("series_rtruediv_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-series-rtruediv-operand-order",
        seed,
        [{"op": "series_rtruediv_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-series-rtruediv-operand-order",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "series_rtruediv_operand_order",
            "source_issue": "https://github.com/pola-rs/polars/issues/17760",
            "expected_series_rtruediv_mismatch": False,
            "expected_series_rtruediv_values": [2.0, 1.5, 4.0 / 3.0],
        },
    )


def generate_polars_reverse_division_columns_case(seed: int) -> Case:
    offset = seed % 3
    rows = [
        {"divisor_value": 1 + offset, "numerator_value": 2 + offset, "group_code": 0},
        {"divisor_value": 2 + offset, "numerator_value": 3 + offset, "group_code": 1},
        {"divisor_value": 3 + offset, "numerator_value": 4 + offset, "group_code": 1},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("divisor_value", "int", nullable=False),
            ColumnSpec("numerator_value", "int", nullable=False),
            ColumnSpec("group_code", "int", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-polars-reverse-division-columns",
        seed,
        [
            {
                "op": "mutate",
                "column": "ratio_value",
                "expr": {
                    "kind": "reverse_division_columns",
                    "source": "divisor_value",
                    "numerator": "numerator_value",
                },
            },
            {"op": "select", "columns": ["group_code", "ratio_value"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-reverse-division-columns",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "polars_reverse_division_columns"},
    )


def generate_pandas_uint64_isin_precision_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("uint64_isin_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-uint64-isin-precision",
        seed,
        [{"op": "uint64_isin_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-uint64-isin-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_uint64_isin_precision",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/59609",
            "expected_uint64_isin_mismatch": False,
            "expected_uint64_isin_value": False,
        },
    )


def generate_duckdb_tuple_anti_null_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("tuple_anti_null_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-tuple-anti-null-semantics",
        seed,
        [{"op": "tuple_anti_null_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-tuple-anti-null-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_tuple_anti_null_semantics",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22418",
            "expected_tuple_anti_null_mismatch": False,
            "expected_tuple_anti_null_counts": [1, 1, 2],
        },
    )


def generate_datafusion_setop_all_duplicate_count_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("setop_all_duplicate_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-datafusion-setop-all-duplicate-count",
        seed,
        [{"op": "setop_all_duplicate_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-datafusion-setop-all-duplicate-count",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "datafusion_setop_all_duplicate_count",
            "source_issue": "https://github.com/apache/datafusion/issues/12956",
            "source_issue_alt": "https://github.com/apache/datafusion/issues/12955",
            "expected_setop_all_duplicate_mismatch": False,
            "expected_except_all_counts": {"a": 1, "b": 1, "c": 2},
            "expected_intersect_all_counts": {"b": 2, "c": 2},
        },
    )


def generate_duckdb_json_predicate_order_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("json_predicate_order_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-json-predicate-order-semantics",
        seed,
        [{"op": "json_predicate_order_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-json-predicate-order-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_json_predicate_order_semantics",
            "source_issue": "https://github.com/duckdb/duckdb/issues/20366",
            "expected_json_predicate_order_mismatch": False,
            "expected_json_predicate_order_rows": 1,
        },
    )


def generate_pandas_sparse_array_mask_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("sparse_mask_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-sparse-array-mask-semantics",
        seed,
        [{"op": "sparse_mask_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-sparse-array-mask-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_sparse_array_mask_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/45284",
            "expected_sparse_mask_mismatch": False,
            "expected_sparse_mask_values": [4.0],
        },
    )


def generate_polars_float_wrap_numerical_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("float_wrap_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-float-wrap-numerical-semantics",
        seed,
        [{"op": "float_wrap_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-float-wrap-numerical-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_float_wrap_numerical_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/18546",
            "expected_float_wrap_mismatch": False,
            "expected_wrapped_uint8": [100, 44],
        },
    )


def generate_pandas_index_bool_result_type_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("index_bool_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-index-bool-result-type",
        seed,
        [{"op": "index_bool_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-index-bool-result-type",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_index_bool_result_type",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/62766",
            "expected_index_bool_mismatch": False,
            "expected_index_bool_type": "Index",
        },
    )


def generate_polars_empty_literal_groupby_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("empty_literal_groupby_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-empty-literal-groupby-semantics",
        seed,
        [{"op": "empty_literal_groupby_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-empty-literal-groupby-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_empty_literal_groupby_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/23870",
            "expected_empty_literal_groupby_mismatch": False,
            "expected_empty_literal_groupby_rows": 0,
        },
    )


def generate_pandas_arrow_string_eq_sum_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_string_eq_sum_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-string-eq-sum-semantics",
        seed,
        [{"op": "arrow_string_eq_sum_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-string-eq-sum-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_string_eq_sum_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63458",
            "expected_arrow_string_eq_sum_mismatch": False,
            "expected_arrow_string_eq_sum_value": 1,
        },
    )


def generate_pandas_arrow_timestamp_loc_slice_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_timestamp_loc_slice_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-timestamp-loc-slice-semantics",
        seed,
        [{"op": "arrow_timestamp_loc_slice_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-timestamp-loc-slice-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_timestamp_loc_slice_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63526",
            "expected_arrow_timestamp_loc_slice_mismatch": False,
            "expected_arrow_timestamp_loc_slice_rows": 1,
        },
    )


def generate_pandas_arrow_timestamp_index_attr_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_timestamp_index_attr_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-timestamp-index-attr-semantics",
        seed,
        [{"op": "arrow_timestamp_index_attr_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-timestamp-index-attr-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_timestamp_index_attr_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63527",
            "expected_arrow_timestamp_index_attr_mismatch": False,
            "expected_arrow_timestamp_index_months": [5, 6],
        },
    )


def generate_pandas_eval_inplace_aliasing_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("eval_inplace_alias_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-eval-inplace-aliasing-semantics",
        seed,
        [{"op": "eval_inplace_alias_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-eval-inplace-aliasing-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_eval_inplace_aliasing_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/65664",
            "expected_eval_inplace_alias_mismatch": False,
            "expected_eval_inplace_alias_rows": 3,
        },
    )


def generate_pandas_bool_reduction_skipna_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("bool_reduction_skipna_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-bool-reduction-skipna-semantics",
        seed,
        [{"op": "bool_reduction_skipna_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-bool-reduction-skipna-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_bool_reduction_skipna_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/65710",
            "expected_bool_reduction_skipna_mismatch": False,
        },
    )


def generate_pyarrow_dataset_isin_all_match_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("dataset_isin_all_match_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-dataset-isin-all-match-semantics",
        seed,
        [{"op": "dataset_isin_all_match_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-dataset-isin-all-match-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_dataset_isin_all_match_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/46183",
            "expected_dataset_isin_all_match_mismatch": False,
            "expected_dataset_isin_all_match_rows": 1,
        },
    )


def generate_pyarrow_run_end_null_compute_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("run_end_null_compute_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-run-end-null-compute-semantics",
        seed,
        [{"op": "run_end_null_compute_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-run-end-null-compute-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_run_end_null_compute_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/49889",
            "expected_run_end_null_compute_mismatch": False,
            "expected_run_end_null_compute_values": [True, None, True, True, True],
        },
    )


def generate_pyarrow_large_string_partition_schema_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("large_string_partition_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-large-string-partition-schema-semantics",
        seed,
        [{"op": "large_string_partition_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-large-string-partition-schema-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_large_string_partition_schema_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/47177",
            "expected_large_string_partition_mismatch": False,
            "expected_large_string_partition_rows": 4,
        },
    )


def generate_pyarrow_hash_pivot_wider_order_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("hash_pivot_wider_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-hash-pivot-wider-order-semantics",
        seed,
        [{"op": "hash_pivot_wider_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-hash-pivot-wider-order-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_hash_pivot_wider_order_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/48679",
            "expected_hash_pivot_wider_mismatch": False,
            "expected_hash_pivot_wider_rows": 3,
        },
    )


def generate_pyarrow_list_flatten_parent_indices_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("list_flatten_parent_indices_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-list-flatten-parent-indices-semantics",
        seed,
        [{"op": "list_flatten_parent_indices_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-list-flatten-parent-indices-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_list_flatten_parent_indices_semantics",
            "expected_list_flatten_parent_indices_mismatch": False,
            "expected_list_flatten_values": [1, 2, None, 3],
            "expected_list_parent_indices": [0, 0, 3, 3],
        },
    )


def generate_polars_rolling_mean_by_null_count_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("rolling_mean_by_null_count_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-rolling-mean-by-null-count-semantics",
        seed,
        [{"op": "rolling_mean_by_null_count_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-rolling-mean-by-null-count-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_rolling_mean_by_null_count_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/27661",
            "expected_rolling_mean_by_null_count_mismatch": False,
            "expected_rolling_mean_by_null_count_values": [None, 400.5, None],
        },
    )


def generate_csv_long_numeric_roundtrip_case(seed: int) -> Case:
    base_values = list(DEFAULT_LONG_NUMERIC_CSV_VALUES)
    rotation = seed % len(base_values)
    values = base_values[rotation:] + base_values[:rotation]
    if seed % 2:
        values.append(str(10**20 + (seed % 997)))
    values = csv_long_numeric_values({"values": values})
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("csv_long_numeric_roundtrip_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-csv-long-numeric-roundtrip",
        seed,
        [{"op": "csv_long_numeric_roundtrip_probe", "as": alias, "values": values}],
    )
    return Case(
        case_id=f"case-{seed:08d}-csv-long-numeric-roundtrip",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "csv_long_numeric_roundtrip",
            "issue_inspiration": "CSV long numeric identifiers should round-trip without lossy inference",
            "expected_csv_long_numeric_roundtrip_mismatch": False,
            "expected_csv_values": values,
        },
    )
