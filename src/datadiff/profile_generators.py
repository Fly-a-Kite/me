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


def generate_case_when_join_key_membership_case(seed: int) -> Case:
    join_kind = "left" if seed % 2 == 0 else "inner"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_key", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "raw_key": " alpha ", "x": 1, "flag": True},
            {"id": 1, "raw_key": "", "x": None, "flag": False},
            {"id": 2, "raw_key": "BETA", "x": 2, "flag": None},
            {"id": 3, "raw_key": None, "x": 3, "flag": True},
            {"id": 4, "raw_key": "space value", "x": -1, "flag": False},
            {"id": 5, "raw_key": "中文", "x": 4, "flag": None},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("weight", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"key_norm": "alpha", "weight": 10, "tag": "hot"},
            {"key_norm": "beta", "weight": None, "tag": "warm"},
            {"key_norm": "space value", "weight": -1, "tag": "space"},
            {"key_norm": "中文", "weight": 5, "tag": "unicode"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-case-when-join-key-membership",
        seed,
        [
            {"op": "mutate", "column": "key_stripped", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {"op": "mutate", "column": "key_norm", "expr": {"kind": "string_lower", "source": "key_stripped"}},
            {"op": "join", "table": "t1", "left_on": "key_norm", "right_on": "key_norm", "how": join_kind},
            {
                "op": "case_when",
                "as": "membership_bucket",
                "condition": {"column": "tag", "cmp": "in_set", "value": ["hot", "space", "unicode"]},
                "then": "dimension_member",
                "else": "dimension_other_or_missing",
            },
            {
                "op": "groupby",
                "keys": ["membership_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "weight", "func": "max", "as": "max_weight"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "membership_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-case-when-join-key-membership",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "case_when_join_key_membership",
            "issue_inspiration": "normalized join key plus case_when membership aggregation",
            "join_kind": join_kind,
        },
    )


def generate_coalesce_union_distinct_type_boundary_case(seed: int) -> Case:
    left_rows = [
        {"id": 0, "g": "alpha", "s": None, "x": 1, "flag": True},
        {"id": 1, "g": None, "s": "fallback", "x": None, "flag": False},
        {"id": 2, "g": "", "s": "", "x": -1, "flag": None},
        {"id": 3, "g": "space value", "s": "space", "x": 2, "flag": True},
    ]
    append_rows = [
        {"id": 4, "g": None, "s": "fallback", "x": 0, "flag": False},
        {"id": 5, "g": "alpha", "s": "ignored", "x": None, "flag": True},
        {"id": 6, "g": "中文", "s": None, "x": 3, "flag": None},
        {"id": 7, "g": "", "s": "empty-left", "x": -1, "flag": False},
    ]
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("s", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    table = TableData("t0", columns, left_rows)
    append = TableData("t_append", columns, append_rows)
    program = Program(
        f"prog-{seed:08d}-coalesce-union-distinct-type-boundary",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "coalesce", "columns": ["g", "s"], "as": "label", "fallback": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "case_when",
                "as": "x_band",
                "condition": {"column": "x", "cmp": "range_closed", "value": [0, 2]},
                "then": "in_band",
                "else": "out_band",
            },
            {"op": "distinct", "columns": ["label", "x_band", "flag", "x"]},
            {
                "op": "groupby",
                "keys": ["label", "x_band"],
                "aggs": [
                    {"column": "flag", "func": "count", "as": "count_flag"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "label", "ascending": True, "nulls": "last"},
                    {"column": "x_band", "ascending": True, "nulls": "last"},
                    {"column": "count_flag", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 8},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-coalesce-union-distinct-type-boundary",
        seed=seed,
        tables=[table, append],
        program=program,
        metadata={
            "generator_profile": "coalesce_union_distinct_type_boundary",
            "issue_inspiration": "coalesce/fill_null through union_all, distinct, groupby, and top-k",
        },
    )


def generate_multi_key_anti_join_null_guard_case(seed: int) -> Case:
    join_kind = "anti_join" if seed % 2 == 0 else "semi_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("k1", "int", nullable=True),
            ColumnSpec("k2", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"row_id": 0, "k1": 1, "k2": "a", "x": 1, "flag": True},
            {"row_id": 1, "k1": 1, "k2": "b", "x": 2, "flag": False},
            {"row_id": 2, "k1": 2, "k2": "a", "x": None, "flag": None},
            {"row_id": 3, "k1": None, "k2": "a", "x": 3, "flag": True},
            {"row_id": 4, "k1": 3, "k2": None, "x": 4, "flag": False},
            {"row_id": 5, "k1": 4, "k2": "space value", "x": -1, "flag": None},
        ],
    )
    right = TableData(
        "t_lookup",
        [
            ColumnSpec("rk1", "int", nullable=False),
            ColumnSpec("rk2", "str", nullable=False),
        ],
        [
            {"rk1": 1, "rk2": "a"},
            {"rk1": 2, "rk2": "a"},
            {"rk1": 9, "rk2": "missing"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-multi-key-anti-join-null-guard",
        seed,
        [
            {"op": "filter", "column": "k1", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "k2", "cmp": "is_not_null", "value": None},
            {
                "op": join_kind,
                "table": "t_lookup",
                "left_on": ["k1", "k2"],
                "right_on": ["rk1", "rk2"],
            },
            {
                "op": "case_when",
                "as": "survivor_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_key",
                "else": "not_true_key",
            },
            {
                "op": "groupby",
                "keys": ["survivor_bucket"],
                "aggs": [
                    {"column": "row_id", "func": "count", "as": "count_rows"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "survivor_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_rows", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-multi-key-anti-join-null-guard",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "multi_key_anti_join_null_guard",
            "issue_inspiration": "multi-key semi/anti join after explicit non-null guards",
            "membership_join_kind": join_kind,
        },
    )


def generate_empty_then_union_groupby_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("s", "str", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "g": "drop", "x": -1, "s": "base"},
            {"id": 1, "g": None, "x": None, "s": None},
            {"id": 2, "g": "drop", "x": 2, "s": ""},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"id": 3, "g": "alpha", "x": 1, "s": "a"},
            {"id": 4, "g": "alpha", "x": None, "s": None},
            {"id": 5, "g": "beta", "x": 2, "s": "b"},
            {"id": 6, "g": None, "x": -1, "s": "missing"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-empty-then-union-groupby",
        seed,
        [
            {"op": "filter", "column": "id", "cmp": "<", "value": 0},
            {"op": "union_all", "table": "t_append"},
            {"op": "fill_null", "column": "g", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "s", "func": "nunique", "as": "uniq_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-empty-then-union-groupby",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "empty_then_union_groupby",
            "issue_inspiration": "empty filtered input preserving schema before union_all and groupby",
        },
    )


def generate_boolean_coalesce_case_membership_case(seed: int) -> Case:
    join_kind = "semi_join" if seed % 2 == 0 else "anti_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "flag": True, "x": 1},
            {"id": 1, "g": "a", "flag": False, "x": 2},
            {"id": 2, "g": "b", "flag": None, "x": None},
            {"id": 3, "g": "b", "flag": True, "x": -1},
            {"id": 4, "g": None, "flag": False, "x": 3},
        ],
    )
    dim = TableData(
        "t_bool_dim",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("dim_flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "dim_flag": None},
            {"id": 1, "dim_flag": True},
            {"id": 2, "dim_flag": False},
            {"id": 9, "dim_flag": True},
        ],
    )
    membership = TableData(
        "t_bool_membership",
        [ColumnSpec("flag_key", "bool", nullable=False)],
        [{"flag_key": True}],
    )
    program = Program(
        f"prog-{seed:08d}-boolean-coalesce-case-membership",
        seed,
        [
            {"op": "join", "table": "t_bool_dim", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "coalesce", "columns": ["dim_flag", "flag"], "as": "flag_effective", "fallback": False},
            {
                "op": join_kind,
                "table": "t_bool_membership",
                "left_on": "flag_effective",
                "right_on": "flag_key",
            },
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag_effective", "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag_effective", "func": "any", "as": "any_effective"},
                    {"column": "flag_effective", "func": "all", "as": "all_effective"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-boolean-coalesce-case-membership",
        seed=seed,
        tables=[left, dim, membership],
        program=program,
        metadata={
            "generator_profile": "boolean_coalesce_case_membership",
            "issue_inspiration": "nullable bool coalesce followed by semi/anti membership and boolean aggregates",
            "membership_join_kind": join_kind,
        },
    )


def generate_numeric_text_cast_membership_aggregation_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("num_s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "num_s": "0", "x": 1, "flag": True},
            {"id": 1, "num_s": "1", "x": 2, "flag": False},
            {"id": 2, "num_s": "-1", "x": None, "flag": None},
            {"id": 3, "num_s": "02", "x": 3, "flag": True},
            {"id": 4, "num_s": "10", "x": -1, "flag": False},
            {"id": 5, "num_s": None, "x": 4, "flag": None},
        ],
    )
    membership = TableData(
        "t_numeric_membership",
        [ColumnSpec("num_value", "int", nullable=False)],
        [{"num_value": -1}, {"num_value": 0}, {"num_value": 2}, {"num_value": 10}],
    )
    program = Program(
        f"prog-{seed:08d}-numeric-text-cast-membership-aggregation",
        seed,
        [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "table": "t_numeric_membership", "left_on": "num_value", "right_on": "num_value"},
            {
                "op": "case_when",
                "as": "num_bucket",
                "condition": {"column": "num_value", "cmp": "in_set", "value": [0, 2]},
                "then": "small_member",
                "else": "other_member",
            },
            {
                "op": "groupby",
                "keys": ["num_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "num_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-numeric-text-cast-membership-aggregation",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "numeric_text_cast_membership_aggregation",
            "issue_inspiration": "integer-string cast feeding membership, case_when, and grouped aggregation",
        },
    )


def generate_string_token_join_distinct_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_label", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "raw_label": " alpha/red ", "x": 1, "flag": True},
            {"id": 1, "raw_label": "alpha blue", "x": None, "flag": False},
            {"id": 2, "raw_label": "beta/red", "x": 2, "flag": None},
            {"id": 3, "raw_label": "", "x": -1, "flag": True},
            {"id": 4, "raw_label": None, "x": 3, "flag": False},
            {"id": 5, "raw_label": "space value/red", "x": 4, "flag": None},
        ],
    )
    right = TableData(
        "t_token_dim",
        [
            ColumnSpec("token", "str", nullable=False),
            ColumnSpec("weight", "int", nullable=True),
        ],
        [
            {"token": "alpha", "weight": 10},
            {"token": "beta", "weight": 20},
            {"token": "space", "weight": None},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-string-token-join-distinct",
        seed,
        [
            {"op": "mutate", "column": "label_clean", "expr": {"kind": "string_strip", "source": "raw_label"}},
            {
                "op": "mutate",
                "column": "label_space",
                "expr": {"kind": "string_replace", "source": "label_clean", "old": "/", "new": " "},
            },
            {
                "op": "mutate",
                "column": "token",
                "expr": {"kind": "string_split_part", "source": "label_space", "sep": " ", "index": 0},
            },
            {"op": "filter", "column": "token", "cmp": "is_not_null", "value": None},
            {"op": "join", "table": "t_token_dim", "left_on": "token", "right_on": "token", "how": "left"},
            {
                "op": "case_when",
                "as": "token_bucket",
                "condition": {"column": "weight", "cmp": "is_not_null", "value": None},
                "then": "matched",
                "else": "unmatched",
            },
            {"op": "distinct", "columns": ["token", "token_bucket", "flag", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "token_bucket", "ascending": True, "nulls": "last"},
                    {"column": "token", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 8},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-token-join-distinct",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "string_token_join_distinct",
            "issue_inspiration": "string normalization and token extraction before join, case_when, distinct, and top-k",
        },
    )


def generate_date_part_row_number_union_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("dt", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "dt": "2024-01-02", "x": 1, "flag": True},
            {"id": 1, "dt": "2024-01-02", "x": 2, "flag": False},
            {"id": 2, "dt": "2025-03-10", "x": None, "flag": None},
            {"id": 3, "dt": None, "x": -1, "flag": True},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"id": 4, "dt": "2024-12-31", "x": 3, "flag": False},
            {"id": 5, "dt": "2025-03-11", "x": 4, "flag": True},
            {"id": 6, "dt": "2026-01-01", "x": None, "flag": None},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-part-row-number-union",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "dt_month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "mutate", "column": "dt_day", "expr": {"kind": "date_part", "source": "dt", "part": "day"}},
            {"op": "filter", "column": "dt_year", "cmp": "is_not_null", "value": None},
            {
                "op": "row_number_filter",
                "partition_by": ["dt_year"],
                "order_by": [
                    {"column": "dt_month", "ascending": True, "nulls": "last"},
                    {"column": "dt_day", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "dt_year", "ascending": True, "nulls": "last"},
                    {"column": "dt_month", "ascending": True, "nulls": "last"},
                    {"column": "dt_day", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "dt_year", "dt_month", "dt_day", "x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-part-row-number-union",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "date_part_row_number_union",
            "issue_inspiration": "date-part extraction after union_all feeding partitioned row-number filtering",
        },
    )


def generate_post_groupby_join_global_aggregate_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"g": "alpha", "x": 1, "flag": True},
            {"g": "alpha", "x": None, "flag": False},
            {"g": "beta", "x": 2, "flag": True},
            {"g": "beta", "x": -1, "flag": None},
            {"g": None, "x": 3, "flag": False},
        ],
    )
    dim = TableData(
        "t_group_dim",
        [
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("category", "str", nullable=True),
        ],
        [
            {"g": "alpha", "category": "kept"},
            {"g": "beta", "category": "kept"},
            {"g": "gamma", "category": "unused"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-post-groupby-join-global-aggregate",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                ],
            },
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {"op": "join", "table": "t_group_dim", "left_on": "g", "right_on": "g", "how": "left"},
            {"op": "filter", "column": "category", "cmp": "==", "value": "kept"},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "sum_x", "func": "sum", "as": "total_sum_x"},
                    {"column": "count_flag", "func": "sum", "as": "total_count_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [{"column": "total_sum_x", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-post-groupby-join-global-aggregate",
        seed=seed,
        tables=[left, dim],
        program=program,
        metadata={
            "generator_profile": "post_groupby_join_global_aggregate",
            "issue_inspiration": "join and global aggregate after grouped aggregation",
        },
    )


def generate_distinct_anti_join_case_topk_case(seed: int) -> Case:
    join_kind = "anti_join" if seed % 2 == 0 else "semi_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1, "flag": True},
            {"id": 1, "g": "a", "x": 1, "flag": True},
            {"id": 2, "g": "b", "x": 2, "flag": False},
            {"id": 3, "g": "b", "x": None, "flag": None},
            {"id": 4, "g": "c", "x": -1, "flag": False},
            {"id": 5, "g": None, "x": 3, "flag": True},
        ],
    )
    membership = TableData(
        "t_g_membership",
        [ColumnSpec("g_key", "str", nullable=False)],
        [{"g_key": "a"}, {"g_key": "missing"}],
    )
    program = Program(
        f"prog-{seed:08d}-distinct-anti-join-case-topk",
        seed,
        [
            {"op": "distinct", "columns": ["g", "x", "flag"]},
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {"op": join_kind, "table": "t_g_membership", "left_on": "g", "right_on": "g_key"},
            {
                "op": "case_when",
                "as": "flag_label",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "not_true",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_label", "ascending": True, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-distinct-anti-join-case-topk",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "distinct_anti_join_case_topk",
            "issue_inspiration": "distinct projection feeding semi/anti membership, case_when, and stable top-k",
            "membership_join_kind": join_kind,
        },
    )


def generate_coalesce_row_number_topk_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("fallback_score", "int", nullable=True),
        ],
        [
            {"id": 0, "grp": "a", "score": 10, "fallback_score": 1},
            {"id": 1, "grp": "a", "score": None, "fallback_score": 5},
            {"id": 2, "grp": "a", "score": 7, "fallback_score": None},
            {"id": 3, "grp": "b", "score": None, "fallback_score": 2},
            {"id": 4, "grp": "b", "score": 3, "fallback_score": 8},
            {"id": 5, "grp": None, "score": 9, "fallback_score": 0},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-coalesce-row-number-topk",
        seed,
        [
            {"op": "filter", "column": "grp", "cmp": "is_not_null", "value": None},
            {"op": "coalesce", "columns": ["score", "fallback_score"], "as": "effective_score", "fallback": 0},
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "effective_score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "effective_score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "grp", "effective_score"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-coalesce-row-number-topk",
        seed=seed,
        tables=[left],
        program=program,
        metadata={
            "generator_profile": "coalesce_row_number_topk",
            "issue_inspiration": "coalesced sort key feeding partitioned top-n row selection",
        },
    )


def generate_union_distinct_anti_running_sum_case(seed: int) -> Case:
    columns = [
        ColumnSpec("sample_id", "int", nullable=False),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("seq", "int", nullable=False),
        ColumnSpec("x", "float", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"sample_id": 0, "grp": "a", "seq": 1, "x": 1.0, "flag": True},
            {"sample_id": 1, "grp": "a", "seq": 2, "x": None, "flag": False},
            {"sample_id": 2, "grp": "b", "seq": 1, "x": 2.5, "flag": None},
            {"sample_id": 3, "grp": "b", "seq": 2, "x": -0.5, "flag": True},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"sample_id": 4, "grp": "a", "seq": 3, "x": 3.0, "flag": None},
            {"sample_id": 5, "grp": "b", "seq": 3, "x": None, "flag": False},
            {"sample_id": 6, "grp": None, "seq": 1, "x": 4.0, "flag": True},
        ],
    )
    excluded = TableData(
        "t_excluded",
        [
            ColumnSpec("grp_key", "str", nullable=False),
            ColumnSpec("seq_key", "int", nullable=False),
        ],
        [
            {"grp_key": "a", "seq_key": 2},
            {"grp_key": "missing", "seq_key": 99},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-union-distinct-anti-running-sum",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "distinct", "columns": ["sample_id", "grp", "seq", "x", "flag"]},
            {"op": "filter", "column": "grp", "cmp": "is_not_null", "value": None},
            {
                "op": "anti_join",
                "table": "t_excluded",
                "left_on": ["grp", "seq"],
                "right_on": ["grp_key", "seq_key"],
            },
            {
                "op": "case_when",
                "as": "flag_score",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": 1,
                "else": 0,
            },
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "sample_id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "float64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "sample_id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["sample_id", "grp", "seq", "flag_score", "run_x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-union-distinct-anti-running-sum",
        seed=seed,
        tables=[base, append, excluded],
        program=program,
        metadata={
            "generator_profile": "union_distinct_anti_running_sum",
            "issue_inspiration": "union/distinct materialization feeding anti-join, case_when, and partitioned running sum",
        },
    )


def generate_null_case_semi_join_groupby_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k", "str", nullable=True),
            ColumnSpec("alt_k", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "k": "a", "alt_k": None, "score": 1, "flag": True},
            {"id": 1, "k": None, "alt_k": "a", "score": None, "flag": False},
            {"id": 2, "k": "b", "alt_k": "fallback", "score": 2, "flag": None},
            {"id": 3, "k": None, "alt_k": "c", "score": 3, "flag": True},
            {"id": 4, "k": "", "alt_k": "empty", "score": None, "flag": None},
            {"id": 5, "k": None, "alt_k": None, "score": -1, "flag": False},
        ],
    )
    membership = TableData(
        "t_member",
        [ColumnSpec("key", "str", nullable=False)],
        [{"key": "a"}, {"key": "b"}, {"key": ""}],
    )
    program = Program(
        f"prog-{seed:08d}-null-case-semi-join-groupby",
        seed,
        [
            {"op": "coalesce", "columns": ["k", "alt_k"], "as": "join_key", "fallback": "missing"},
            {
                "op": "case_when",
                "as": "score_state",
                "condition": {"column": "score", "cmp": "is_null", "value": None},
                "then": "missing_score",
                "else": "scored",
            },
            {"op": "semi_join", "table": "t_member", "left_on": "join_key", "right_on": "key"},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "groupby",
                "keys": ["join_key", "score_state"],
                "aggs": [
                    {"column": "score", "func": "sum", "as": "sum_score"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "join_key", "ascending": True, "nulls": "last"},
                    {"column": "score_state", "ascending": True, "nulls": "last"},
                    {"column": "sum_score", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-case-semi-join-groupby",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "null_case_semi_join_groupby",
            "issue_inspiration": "null coalesce and case_when feeding semi-join membership and boolean aggregation",
        },
    )


def generate_date_string_cast_row_number_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("dt", "str", nullable=True),
            ColumnSpec("num_s", "str", nullable=True),
            ColumnSpec("label", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "dt": "2024-01-02", "num_s": "1.5", "label": " Alpha ", "flag": True},
            {"id": 1, "dt": "2024-01-03", "num_s": "2.0", "label": "alpha", "flag": False},
            {"id": 2, "dt": "2024-02-01", "num_s": None, "label": "", "flag": None},
            {"id": 3, "dt": "2025-01-01", "num_s": "-1.25", "label": "Beta", "flag": True},
            {"id": 4, "dt": "2025-01-02", "num_s": "10.0", "label": None, "flag": False},
            {"id": 5, "dt": None, "num_s": "3.5", "label": "gamma", "flag": None},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-string-cast-row-number",
        seed,
        [
            {"op": "mutate", "column": "label_clean", "expr": {"kind": "string_strip", "source": "label"}},
            {"op": "mutate", "column": "label_key", "expr": {"kind": "string_lower", "source": "label_clean"}},
            {
                "op": "mutate",
                "column": "amount",
                "expr": {"kind": "cast", "source": "num_s", "to": "float", "input_domain": "numeric_string"},
            },
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "filter", "column": "amount", "cmp": "is_not_null", "value": None},
            {
                "op": "row_number_filter",
                "partition_by": ["dt_year"],
                "order_by": [
                    {"column": "amount", "ascending": False, "nulls": "last"},
                    {"column": "label_key", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "dt_year", "ascending": True, "nulls": "last"},
                    {"column": "amount", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "dt_year", "label_key", "amount", "flag"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-string-cast-row-number",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "date_string_cast_row_number",
            "issue_inspiration": "date extraction, numeric-string casts, and normalized string order in row-number top-k",
        },
    )


def generate_drop_nulls_coalesce_distinct_join_topk_case(seed: int) -> Case:
    join_kind = "left" if seed % 2 == 0 else "inner"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k", "str", nullable=True),
            ColumnSpec("alt_k", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "k": "alpha", "alt_k": None, "x": 1, "flag": True},
            {"id": 1, "k": None, "alt_k": "alpha", "x": None, "flag": False},
            {"id": 2, "k": "", "alt_k": "empty", "x": -1, "flag": None},
            {"id": 3, "k": None, "alt_k": None, "x": 2, "flag": True},
            {"id": 4, "k": "space value", "alt_k": "space", "x": None, "flag": False},
            {"id": 5, "k": "中文", "alt_k": None, "x": 3, "flag": None},
        ],
    )
    dim = TableData(
        "t_key_dim",
        [
            ColumnSpec("join_key", "str", nullable=False),
            ColumnSpec("weight", "int", nullable=True),
            ColumnSpec("label", "str", nullable=True),
        ],
        [
            {"join_key": "alpha", "weight": 10, "label": "hot"},
            {"join_key": "", "weight": None, "label": "empty"},
            {"join_key": "space value", "weight": 5, "label": "space"},
            {"join_key": "中文", "weight": 7, "label": "unicode"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-drop-nulls-coalesce-distinct-join-topk",
        seed,
        [
            {"op": "coalesce", "columns": ["k", "alt_k"], "as": "join_key", "fallback": "missing"},
            {"op": "drop_nulls", "columns": ["join_key"]},
            {"op": "distinct", "columns": ["join_key", "x", "flag"]},
            {"op": "join", "table": "t_key_dim", "left_on": "join_key", "right_on": "join_key", "how": join_kind},
            {
                "op": "select",
                "columns": ["join_key", "x", "flag", "weight", "label"],
            },
            {"op": "drop_nulls", "columns": ["weight"]},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "case_when",
                "as": "match_bucket",
                "condition": {"column": "weight", "cmp": "is_not_null", "value": None},
                "then": "dimension_match",
                "else": "dimension_missing",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "match_bucket", "ascending": True, "nulls": "last"},
                    {"column": "join_key", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 8},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-drop-nulls-coalesce-distinct-join-topk",
        seed=seed,
        tables=[left, dim],
        program=program,
        metadata={
            "generator_profile": "drop_nulls_coalesce_distinct_join_topk",
            "issue_inspiration": "drop-null cleanup after coalesce feeding distinct, join, fill_null, case_when, and top-k",
            "join_kind": join_kind,
        },
    )


def generate_date_part_distinct_offset_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("dt", "str", nullable=True),
        ColumnSpec("bucket", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "dt": "2024-01-02", "bucket": "a", "x": 1},
            {"id": 1, "dt": "2024-01-31", "bucket": "a", "x": None},
            {"id": 2, "dt": "2024-02-01", "bucket": "b", "x": 2},
            {"id": 3, "dt": None, "bucket": "missing-date", "x": -1},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"id": 4, "dt": "2025-01-02", "bucket": "a", "x": 3},
            {"id": 5, "dt": "2025-01-02", "bucket": "a", "x": 3},
            {"id": 6, "dt": "2025-03-15", "bucket": None, "x": None},
            {"id": 7, "dt": "2026-12-31", "bucket": "z", "x": 5},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-part-distinct-offset",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "dt_month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "filter", "column": "dt_year", "cmp": "is_not_null", "value": None},
            {"op": "fill_null", "column": "bucket", "value": "missing"},
            {"op": "distinct", "columns": ["dt_year", "dt_month", "bucket", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "dt_year", "ascending": True, "nulls": "last"},
                    {"column": "dt_month", "ascending": False, "nulls": "last"},
                    {"column": "bucket", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": seed % 3},
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-part-distinct-offset",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "date_part_distinct_offset",
            "issue_inspiration": "date-part extraction after union_all feeding fill_null, distinct, offset, and top-k",
        },
    )


def generate_bool_fill_null_membership_row_number_case(seed: int) -> Case:
    join_kind = "semi_join" if seed % 2 == 0 else "anti_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "grp": "a", "score": 10, "flag": True},
            {"id": 1, "grp": "a", "score": None, "flag": None},
            {"id": 2, "grp": "a", "score": 5, "flag": False},
            {"id": 3, "grp": "b", "score": 7, "flag": True},
            {"id": 4, "grp": "b", "score": None, "flag": False},
            {"id": 5, "grp": None, "score": 3, "flag": None},
        ],
    )
    membership = TableData(
        "t_bucket_member",
        [ColumnSpec("flag_bucket", "str", nullable=False)],
        [{"flag_bucket": "true_flag"}, {"flag_bucket": "false_or_missing"}],
    )
    program = Program(
        f"prog-{seed:08d}-bool-fill-null-membership-row-number",
        seed,
        [
            {"op": "fill_null", "column": "flag", "value": False},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "false_or_missing",
            },
            {"op": join_kind, "table": "t_bucket_member", "left_on": "flag_bucket", "right_on": "flag_bucket"},
            {
                "op": "row_number_filter",
                "partition_by": ["flag_bucket"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "score", "func": "sum", "as": "sum_score"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "sum_score", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-fill-null-membership-row-number",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "bool_fill_null_membership_row_number",
            "issue_inspiration": "boolean fill_null and case_when feeding membership, row-number top-n, and aggregation",
            "membership_join_kind": join_kind,
        },
    )


def generate_string_numeric_cast_anti_join_aggregate_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_key", "str", nullable=True),
            ColumnSpec("num_s", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "raw_key": " Alpha ", "num_s": "1", "flag": True},
            {"id": 1, "raw_key": "alpha", "num_s": "2", "flag": False},
            {"id": 2, "raw_key": "Beta", "num_s": None, "flag": None},
            {"id": 3, "raw_key": "", "num_s": "-1", "flag": True},
            {"id": 4, "raw_key": "中文", "num_s": "10", "flag": False},
            {"id": 5, "raw_key": None, "num_s": "0", "flag": None},
        ],
    )
    excluded = TableData(
        "t_excluded_key",
        [ColumnSpec("key_norm", "str", nullable=False)],
        [{"key_norm": "alpha"}, {"key_norm": ""}],
    )
    program = Program(
        f"prog-{seed:08d}-string-numeric-cast-anti-join-aggregate",
        seed,
        [
            {"op": "mutate", "column": "key_clean", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {"op": "mutate", "column": "key_norm", "expr": {"kind": "string_lower", "source": "key_clean"}},
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "key_norm", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "anti_join", "table": "t_excluded_key", "left_on": "key_norm", "right_on": "key_norm"},
            {
                "op": "groupby",
                "keys": ["key_norm"],
                "aggs": [
                    {"column": "num_value", "func": "sum", "as": "sum_num"},
                    {"column": "num_value", "func": "nunique", "as": "uniq_num"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_num", "ascending": False, "nulls": "last"},
                    {"column": "key_norm", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-numeric-cast-anti-join-aggregate",
        seed=seed,
        tables=[left, excluded],
        program=program,
        metadata={
            "generator_profile": "string_numeric_cast_anti_join_aggregate",
            "issue_inspiration": "string normalization and integer-string casts feeding anti-join and grouped aggregation",
        },
    )


def generate_post_aggregate_case_membership_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"grp": "a", "x": 1, "flag": True},
            {"grp": "a", "x": 2, "flag": None},
            {"grp": "b", "x": None, "flag": False},
            {"grp": "b", "x": -1, "flag": True},
            {"grp": "c", "x": 0, "flag": None},
            {"grp": None, "x": 5, "flag": False},
        ],
    )
    membership = TableData(
        "t_group_class",
        [ColumnSpec("group_class", "str", nullable=False)],
        [{"group_class": "large_group"}, {"group_class": "small_group"}],
    )
    program = Program(
        f"prog-{seed:08d}-post-aggregate-case-membership",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                ],
            },
            {"op": "filter", "column": "grp", "cmp": "is_not_null", "value": None},
            {
                "op": "case_when",
                "as": "group_class",
                "condition": {"column": "count_flag", "cmp": ">=", "value": 1},
                "then": "large_group",
                "else": "small_group",
            },
            {"op": "semi_join", "table": "t_group_class", "left_on": "group_class", "right_on": "group_class"},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "sum_x", "func": "sum", "as": "total_sum_x"},
                    {"column": "count_flag", "func": "sum", "as": "total_count_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "total_count_flag", "ascending": False, "nulls": "last"},
                    {"column": "total_sum_x", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-post-aggregate-case-membership",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "post_aggregate_case_membership",
            "issue_inspiration": "grouped aggregate output classified by case_when, filtered through membership, then globally aggregated",
        },
    )


def generate_multi_key_nullable_membership_window_case(seed: int) -> Case:
    join_kind = "semi_join" if seed % 2 == 0 else "anti_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k1", "str", nullable=True),
            ColumnSpec("k1_alt", "str", nullable=True),
            ColumnSpec("k2", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "k1": "alpha", "k1_alt": None, "k2": 1, "score": 10, "flag": True},
            {"id": 1, "k1": "alpha", "k1_alt": "alpha", "k2": None, "score": None, "flag": False},
            {"id": 2, "k1": None, "k1_alt": "missing", "k2": 1, "score": 5, "flag": None},
            {"id": 3, "k1": "beta", "k1_alt": None, "k2": 2, "score": 7, "flag": True},
            {"id": 4, "k1": "beta", "k1_alt": "beta", "k2": 2, "score": None, "flag": False},
            {"id": 5, "k1": "", "k1_alt": "empty", "k2": 0, "score": -2, "flag": None},
            {"id": 6, "k1": "space value", "k1_alt": None, "k2": -1, "score": 3, "flag": True},
        ],
    )
    membership = TableData(
        "t_multi_key_member",
        [
            ColumnSpec("k1_norm", "str", nullable=False),
            ColumnSpec("k2_norm", "int", nullable=False),
        ],
        [
            {"k1_norm": "alpha", "k2_norm": 1},
            {"k1_norm": "missing", "k2_norm": 1},
            {"k1_norm": "beta", "k2_norm": 2},
            {"k1_norm": "", "k2_norm": 0},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-multi-key-nullable-membership-window",
        seed,
        [
            {"op": "coalesce", "columns": ["k1", "k1_alt"], "as": "k1_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "k2", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "false_or_missing",
            },
            {
                "op": join_kind,
                "table": "t_multi_key_member",
                "left_on": ["k1_norm", "k2"],
                "right_on": ["k1_norm", "k2_norm"],
            },
            {
                "op": "row_number_filter",
                "partition_by": ["flag_bucket", "k1_norm"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket", "k1_norm"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "score", "func": "sum", "as": "sum_score"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "k1_norm", "ascending": True, "nulls": "last"},
                    {"column": "sum_score", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-multi-key-nullable-membership-window",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "multi_key_nullable_membership_window",
            "issue_inspiration": "nullable multi-key normalization feeding semi/anti membership, row-number, and grouped aggregation",
            "membership_join_kind": join_kind,
        },
    )


def generate_string_empty_pattern_membership_distinct_case(seed: int) -> Case:
    join_kind = "semi_join" if seed % 2 == 0 else "anti_join"
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_key", "str", nullable=True),
            ColumnSpec("fallback_key", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "raw_key": " Alpha:1 ", "fallback_key": "fallback-a", "x": 10},
            {"id": 1, "raw_key": "alpha:2", "fallback_key": None, "x": None},
            {"id": 2, "raw_key": "", "fallback_key": "empty-fallback", "x": 3},
            {"id": 3, "raw_key": "Beta", "fallback_key": "beta", "x": -1},
            {"id": 4, "raw_key": None, "fallback_key": "missing", "x": 4},
            {"id": 5, "raw_key": "space value:tail", "fallback_key": None, "x": None},
        ],
    )
    membership = TableData(
        "t_string_member",
        [ColumnSpec("lookup_key", "str", nullable=False)],
        [{"lookup_key": "alpha"}, {"lookup_key": "beta"}, {"lookup_key": "missing"}],
    )
    program = Program(
        f"prog-{seed:08d}-string-empty-pattern-membership-distinct",
        seed,
        [
            {"op": "mutate", "column": "key_clean", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {
                "op": "mutate",
                "column": "token",
                "expr": {"kind": "string_split_part", "source": "key_clean", "sep": ":", "index": 0},
            },
            {"op": "mutate", "column": "token_nonempty", "expr": {"kind": "string_null_if_empty", "source": "token"}},
            {"op": "mutate", "column": "token_norm", "expr": {"kind": "string_lower", "source": "token_nonempty"}},
            {"op": "coalesce", "columns": ["token_norm", "fallback_key"], "as": "lookup_key", "fallback": "missing"},
            {"op": "mutate", "column": "has_space", "expr": {"kind": "string_contains", "source": "key_clean", "needle": " "}},
            {
                "op": "case_when",
                "as": "shape_bucket",
                "condition": {"column": "has_space", "cmp": "bool_is_true", "value": None},
                "then": "contains_space",
                "else": "compact",
            },
            {"op": join_kind, "table": "t_string_member", "left_on": "lookup_key", "right_on": "lookup_key"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "distinct", "columns": ["lookup_key", "shape_bucket", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "lookup_key", "ascending": True, "nulls": "last"},
                    {"column": "shape_bucket", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-empty-pattern-membership-distinct",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "string_empty_pattern_membership_distinct",
            "issue_inspiration": "string strip/split/null-if-empty and pattern predicates feeding membership, fill_null, distinct, and top-k",
            "membership_join_kind": join_kind,
        },
    )


def generate_date_cast_union_running_sum_topk_case(seed: int) -> Case:
    columns = [
        ColumnSpec("seq", "int", nullable=False),
        ColumnSpec("account", "str", nullable=True),
        ColumnSpec("dt", "str", nullable=True),
        ColumnSpec("amount_s", "str", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"seq": 0, "account": "a", "dt": "2024-01-02", "amount_s": "10"},
            {"seq": 1, "account": "a", "dt": "2024-01-03", "amount_s": None},
            {"seq": 2, "account": "b", "dt": "2024-02-01", "amount_s": "-5"},
            {"seq": 3, "account": None, "dt": "2024-02-02", "amount_s": "7"},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"seq": 4, "account": "a", "dt": "2025-01-01", "amount_s": "3"},
            {"seq": 5, "account": "b", "dt": None, "amount_s": "11"},
            {"seq": 6, "account": "b", "dt": "2025-03-15", "amount_s": "0"},
            {"seq": 7, "account": "c", "dt": "2026-12-31", "amount_s": "-2"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-cast-union-running-sum-topk",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "dt_month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {
                "op": "mutate",
                "column": "amount_i",
                "expr": {"kind": "cast", "source": "amount_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "fill_null", "column": "account", "value": "missing"},
            {"op": "fill_null", "column": "amount_i", "value": 0},
            {"op": "distinct", "columns": ["seq", "account", "dt_year", "dt_month", "amount_i"]},
            {"op": "filter", "column": "dt_year", "cmp": "is_not_null", "value": None},
            {
                "op": "running_sum",
                "source": "amount_i",
                "column": "run_amount",
                "partition_by": ["account"],
                "order_by": [
                    {"column": "dt_year", "ascending": True, "nulls": "last"},
                    {"column": "dt_month", "ascending": True, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["account"],
                "order_by": [
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "account", "ascending": True, "nulls": "last"},
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["seq", "account", "dt_year", "dt_month", "amount_i", "run_amount"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-cast-union-running-sum-topk",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "date_cast_union_running_sum_topk",
            "issue_inspiration": "date-part extraction and integer-string casts after union feeding distinct, running sum, and per-group top-k",
        },
    )


def generate_coalesce_anti_join_union_topk_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("k_primary", "str", nullable=True),
        ColumnSpec("k_fallback", "str", nullable=True),
        ColumnSpec("bucket", "int", nullable=True),
        ColumnSpec("score", "int", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "k_primary": "alpha", "k_fallback": None, "bucket": 1, "score": 10, "flag": True},
            {"id": 1, "k_primary": None, "k_fallback": "alpha", "bucket": None, "score": None, "flag": False},
            {"id": 2, "k_primary": "beta", "k_fallback": "b", "bucket": 2, "score": 5, "flag": None},
            {"id": 3, "k_primary": "", "k_fallback": "empty", "bucket": 0, "score": -1, "flag": True},
        ],
    )
    append = TableData(
        "t_append",
        columns,
        [
            {"id": 4, "k_primary": None, "k_fallback": "missing", "bucket": -1, "score": 7, "flag": None},
            {"id": 5, "k_primary": "alpha", "k_fallback": "other", "bucket": 1, "score": 3, "flag": True},
            {"id": 6, "k_primary": "space value", "k_fallback": None, "bucket": None, "score": None, "flag": False},
            {"id": 7, "k_primary": "gamma", "k_fallback": "g", "bucket": 3, "score": 0, "flag": None},
        ],
    )
    blocked = TableData(
        "t_blocked_keys",
        [
            ColumnSpec("k_norm", "str", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"k_norm": "alpha", "bucket_norm": 1},
            {"k_norm": "missing", "bucket_norm": -1},
            {"k_norm": "", "bucket_norm": 0},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-coalesce-anti-join-union-topk",
        seed,
        [
            {"op": "union_all", "table": "t_append"},
            {"op": "coalesce", "columns": ["k_primary", "k_fallback"], "as": "k_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_not_false", "value": None},
                "then": "true_or_missing",
                "else": "false_only",
            },
            {
                "op": "anti_join",
                "table": "t_blocked_keys",
                "left_on": ["k_norm", "bucket"],
                "right_on": ["k_norm", "bucket_norm"],
            },
            {"op": "distinct", "columns": ["k_norm", "bucket", "flag_bucket", "score"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "k_norm", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-coalesce-anti-join-union-topk",
        seed=seed,
        tables=[base, append, blocked],
        program=program,
        metadata={
            "generator_profile": "coalesce_anti_join_union_topk",
            "issue_inspiration": "coalesced nullable keys and filled numeric keys after union feeding multi-key anti membership and distinct top-k",
        },
    )


def generate_bool_null_distinct_running_sum_case(seed: int) -> Case:
    rows = [
        {"id": 0, "grp": "a", "seq": 0, "flag": True, "x": 1},
        {"id": 1, "grp": "a", "seq": 1, "flag": None, "x": None},
        {"id": 2, "grp": "a", "seq": 1, "flag": False, "x": 2},
        {"id": 3, "grp": "b", "seq": 0, "flag": None, "x": -1},
        {"id": 4, "grp": "b", "seq": 2, "flag": True, "x": None},
        {"id": 5, "grp": None, "seq": 3, "flag": False, "x": 3},
        {"id": 6, "grp": None, "seq": 4, "flag": None, "x": 0},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("seq", "int", nullable=False),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-bool-null-distinct-running-sum",
        seed,
        [
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "case_when",
                "as": "truth_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_not_false", "value": None},
                "then": "true_or_null",
                "else": "false_only",
            },
            {"op": "distinct", "columns": ["grp", "seq", "truth_bucket", "x"]},
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["grp", "truth_bucket"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["truth_bucket"],
                "order_by": [
                    {"column": "run_x", "ascending": False, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 3,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "truth_bucket", "ascending": True, "nulls": "last"},
                    {"column": "run_x", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-null-distinct-running-sum",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bool_null_distinct_running_sum",
            "issue_inspiration": "three-valued boolean classification through distinct and partitioned running sum with tied order keys",
        },
    )


def generate_date_string_membership_offset_window_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_key", "str", nullable=True),
            ColumnSpec("fallback_key", "str", nullable=True),
            ColumnSpec("dt", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
        ],
        [
            {"id": 0, "raw_key": " Alpha/2024 ", "fallback_key": None, "dt": "2024-01-01", "score": 5},
            {"id": 1, "raw_key": "alpha/2024", "fallback_key": "alpha", "dt": "2024-01-02", "score": None},
            {"id": 2, "raw_key": "", "fallback_key": "missing", "dt": None, "score": 3},
            {"id": 3, "raw_key": "Beta/2025", "fallback_key": None, "dt": "2025-02-03", "score": 7},
            {"id": 4, "raw_key": None, "fallback_key": "gamma", "dt": "2026-03-04", "score": -1},
            {"id": 5, "raw_key": "space value/2025", "fallback_key": None, "dt": "2025-04-05", "score": None},
        ],
    )
    membership = TableData(
        "t_date_string_member",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("year", "int", nullable=False),
        ],
        [
            {"key_norm": "alpha", "year": 2024},
            {"key_norm": "beta", "year": 2025},
            {"key_norm": "space value", "year": 2025},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-string-membership-offset-window",
        seed,
        [
            {"op": "mutate", "column": "key_clean", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {
                "op": "mutate",
                "column": "key_token",
                "expr": {"kind": "string_split_part", "source": "key_clean", "sep": "/", "index": 0},
            },
            {"op": "mutate", "column": "key_nonempty", "expr": {"kind": "string_null_if_empty", "source": "key_token"}},
            {"op": "mutate", "column": "key_lower", "expr": {"kind": "string_lower", "source": "key_nonempty"}},
            {"op": "coalesce", "columns": ["key_lower", "fallback_key"], "as": "key_norm", "fallback": "missing"},
            {"op": "mutate", "column": "year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "semi_join",
                "table": "t_date_string_member",
                "left_on": ["key_norm", "year"],
                "right_on": ["key_norm", "year"],
            },
            {
                "op": "row_number_filter",
                "partition_by": ["year"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "year", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "key_norm", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": seed % 2},
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-string-membership-offset-window",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "date_string_membership_offset_window",
            "issue_inspiration": "normalized string/date keys feeding multi-key membership, per-year row-number, and offset/limit top-k",
        },
    )


def generate_empty_union_window_aggregate_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("seq", "int", nullable=False),
        ColumnSpec("amount", "int", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "grp": "drop", "seq": 0, "amount": 99, "flag": True},
            {"id": 1, "grp": None, "seq": 1, "amount": None, "flag": None},
        ],
    )
    append = TableData(
        "t_append_empty_window",
        columns,
        [
            {"id": 2, "grp": "a", "seq": 0, "amount": 1, "flag": True},
            {"id": 3, "grp": "a", "seq": 1, "amount": None, "flag": False},
            {"id": 4, "grp": "b", "seq": 0, "amount": -2, "flag": None},
            {"id": 5, "grp": None, "seq": 2, "amount": 3, "flag": True},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-empty-union-window-aggregate",
        seed,
        [
            {"op": "filter", "column": "id", "cmp": "<", "value": 0},
            {"op": "union_all", "table": "t_append_empty_window"},
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "amount", "value": 0},
            {
                "op": "running_sum",
                "source": "amount",
                "column": "run_amount",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "run_amount", "func": "max", "as": "max_run_amount"},
                    {"column": "flag", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_run_amount", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-empty-union-window-aggregate",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "empty_union_window_aggregate",
            "issue_inspiration": "schema-preserving empty input before union feeding window and grouped aggregate",
        },
    )


def generate_duplicate_key_join_distinct_anti_topk_case(seed: int) -> Case:
    join_kind = "left" if seed % 2 == 0 else "inner"
    left = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("key", "str", nullable=True),
            ColumnSpec("key_alt", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
        ],
        [
            {"row_id": 0, "key": "a", "key_alt": None, "bucket": 1, "score": 5},
            {"row_id": 1, "key": "a", "key_alt": "unused", "bucket": 1, "score": None},
            {"row_id": 2, "key": "b", "key_alt": None, "bucket": 2, "score": 7},
            {"row_id": 3, "key": None, "key_alt": "missing", "bucket": None, "score": -1},
            {"row_id": 4, "key": "", "key_alt": "empty", "bucket": 0, "score": 0},
        ],
    )
    dim = TableData(
        "t_dup_dim",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("dim_rank", "int", nullable=True),
            ColumnSpec("label", "str", nullable=True),
        ],
        [
            {"key_norm": "a", "dim_rank": 10, "label": "first"},
            {"key_norm": "a", "dim_rank": 20, "label": "second"},
            {"key_norm": "b", "dim_rank": None, "label": "only"},
            {"key_norm": "missing", "dim_rank": -1, "label": None},
        ],
    )
    blocked = TableData(
        "t_dup_blocked",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"key_norm": "a", "bucket_norm": 1},
            {"key_norm": "missing", "bucket_norm": -1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-duplicate-key-join-distinct-anti-topk",
        seed,
        [
            {"op": "coalesce", "columns": ["key", "key_alt"], "as": "key_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {"op": "join", "table": "t_dup_dim", "left_on": "key_norm", "right_on": "key_norm", "how": join_kind},
            {
                "op": "case_when",
                "as": "rank_bucket",
                "condition": {"column": "dim_rank", "cmp": "range_closed", "value": [0, 15]},
                "then": "rank_low",
                "else": "rank_other_or_missing",
            },
            {"op": "distinct", "columns": ["key_norm", "bucket", "score", "rank_bucket"]},
            {
                "op": "anti_join",
                "table": "t_dup_blocked",
                "left_on": ["key_norm", "bucket"],
                "right_on": ["key_norm", "bucket_norm"],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "rank_bucket", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "key_norm", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-duplicate-key-join-distinct-anti-topk",
        seed=seed,
        tables=[left, dim, blocked],
        program=program,
        metadata={
            "generator_profile": "duplicate_key_join_distinct_anti_topk",
            "issue_inspiration": "duplicate dimension keys expanding join cardinality before distinct, multi-key anti membership, and top-k",
            "join_kind": join_kind,
        },
    )


def generate_large_int_text_membership_window_case(seed: int) -> Case:
    high = 9_007_199_254_740_992
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("acct", "str", nullable=True),
            ColumnSpec("num_s", "str", nullable=True),
            ColumnSpec("delta", "int", nullable=True),
        ],
        [
            {"id": 0, "acct": "a", "num_s": str(high - 1), "delta": 1},
            {"id": 1, "acct": "a", "num_s": str(high), "delta": None},
            {"id": 2, "acct": "a", "num_s": str(high + 1), "delta": -1},
            {"id": 3, "acct": "b", "num_s": str(-high), "delta": 2},
            {"id": 4, "acct": None, "num_s": "0", "delta": 3},
            {"id": 5, "acct": "b", "num_s": None, "delta": None},
        ],
    )
    membership = TableData(
        "t_large_int_member",
        [ColumnSpec("num_value", "int", nullable=False)],
        [
            {"num_value": high - 1},
            {"num_value": high + 1},
            {"num_value": -high},
            {"num_value": 0},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-large-int-text-membership-window",
        seed,
        [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "table": "t_large_int_member", "left_on": "num_value", "right_on": "num_value"},
            {"op": "fill_null", "column": "acct", "value": "missing"},
            {"op": "fill_null", "column": "delta", "value": 0},
            {
                "op": "case_when",
                "as": "magnitude_bucket",
                "condition": {"column": "num_value", "cmp": "range_closed", "value": [-high, high]},
                "then": "within_exact_boundary",
                "else": "outside_exact_boundary",
            },
            {
                "op": "running_sum",
                "source": "delta",
                "column": "run_delta",
                "partition_by": ["acct", "magnitude_bucket"],
                "order_by": [
                    {"column": "num_value", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "magnitude_bucket", "ascending": True, "nulls": "last"},
                    {"column": "num_value", "ascending": True, "nulls": "last"},
                    {"column": "acct", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-large-int-text-membership-window",
        seed=seed,
        tables=[left, membership],
        program=program,
        metadata={
            "generator_profile": "large_int_text_membership_window",
            "issue_inspiration": "integer-string cast beyond IEEE-754 exactness feeding membership, case_when, and window order keys",
        },
    )


def generate_nested_topk_offset_aggregate_case(seed: int) -> Case:
    rows = [
        {"id": 0, "grp": "a", "score": 10, "x": 1, "flag": True},
        {"id": 1, "grp": "a", "score": 10, "x": None, "flag": False},
        {"id": 2, "grp": "b", "score": 7, "x": 2, "flag": None},
        {"id": 3, "grp": "b", "score": 7, "x": -1, "flag": True},
        {"id": 4, "grp": "c", "score": None, "x": 3, "flag": False},
        {"id": 5, "grp": None, "score": 5, "x": None, "flag": None},
        {"id": 6, "grp": "d", "score": -1, "x": 4, "flag": True},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-nested-topk-offset-aggregate",
        seed,
        [
            {"op": "fill_null", "column": "score", "value": -999},
            {
                "op": "sort",
                "keys": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 6},
            {"op": "offset", "n": seed % 3},
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 4},
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "score", "func": "max", "as": "max_score"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_score", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-nested-topk-offset-aggregate",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "nested_topk_offset_aggregate",
            "issue_inspiration": "nested limit/offset/top-k ordering feeding grouped aggregate",
        },
    )


def generate_union_distinct_empty_string_window_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("raw_key", "str", nullable=True),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("amount", "int", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "raw_key": " Alpha ", "grp": "g1", "amount": 5, "flag": True},
            {"id": 1, "raw_key": "alpha", "grp": "g1", "amount": None, "flag": False},
            {"id": 2, "raw_key": "", "grp": None, "amount": -2, "flag": None},
            {"id": 3, "raw_key": None, "grp": "g2", "amount": 3, "flag": True},
        ],
    )
    empty = TableData("t_v8_empty_union", columns, [])
    append = TableData(
        "t_v8_string_append",
        columns,
        [
            {"id": 4, "raw_key": "BETA", "grp": "g2", "amount": 7, "flag": False},
            {"id": 5, "raw_key": " beta ", "grp": "g2", "amount": None, "flag": None},
            {"id": 6, "raw_key": "space value", "grp": None, "amount": 1, "flag": True},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-union-distinct-empty-string-window",
        seed,
        [
            {"op": "union_all", "table": "t_v8_empty_union"},
            {"op": "union_all", "table": "t_v8_string_append"},
            {"op": "mutate", "column": "key_strip", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {"op": "mutate", "column": "key_lower", "expr": {"kind": "string_lower", "source": "key_strip"}},
            {"op": "mutate", "column": "key_nonempty", "expr": {"kind": "string_null_if_empty", "source": "key_lower"}},
            {"op": "coalesce", "columns": ["key_nonempty", "grp"], "as": "key_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "amount", "value": 0},
            {"op": "distinct", "columns": ["key_norm", "grp", "amount", "flag"]},
            {
                "op": "running_sum",
                "source": "amount",
                "column": "run_amount",
                "partition_by": ["key_norm"],
                "order_by": [
                    {"column": "amount", "ascending": True, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["key_norm"],
                "order_by": [
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "key_norm", "ascending": True, "nulls": "last"},
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-union-distinct-empty-string-window",
        seed=seed,
        tables=[base, empty, append],
        program=program,
        metadata={
            "generator_profile": "union_distinct_empty_string_window",
            "issue_inspiration": "empty union branch plus string normalization and distinct before partitioned running window",
        },
    )


def generate_multi_key_semi_join_window_aggregate_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k1", "str", nullable=True),
            ColumnSpec("k2", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "k1": "a", "k2": 1, "score": 10, "flag": True},
            {"id": 1, "k1": "a", "k2": 1, "score": None, "flag": False},
            {"id": 2, "k1": "b", "k2": 2, "score": 5, "flag": None},
            {"id": 3, "k1": None, "k2": 2, "score": 7, "flag": True},
            {"id": 4, "k1": "", "k2": None, "score": -1, "flag": False},
            {"id": 5, "k1": "c", "k2": 3, "score": 5, "flag": None},
        ],
    )
    allow = TableData(
        "t_v8_allow",
        [
            ColumnSpec("k1_norm", "str", nullable=False),
            ColumnSpec("k2_norm", "int", nullable=False),
        ],
        [
            {"k1_norm": "a", "k2_norm": 1},
            {"k1_norm": "b", "k2_norm": 2},
            {"k1_norm": "missing", "k2_norm": 2},
            {"k1_norm": "missing", "k2_norm": -1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-multi-key-semi-join-window-aggregate",
        seed,
        [
            {"op": "mutate", "column": "k1_nonempty", "expr": {"kind": "string_null_if_empty", "source": "k1"}},
            {"op": "coalesce", "columns": ["k1_nonempty", "k1"], "as": "k1_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "k2", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "semi_join",
                "table": "t_v8_allow",
                "left_on": ["k1_norm", "k2"],
                "right_on": ["k1_norm", "k2_norm"],
            },
            {"op": "distinct", "columns": ["k1_norm", "k2", "score", "flag"]},
            {
                "op": "row_number_filter",
                "partition_by": ["k1_norm", "k2"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "flag", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["k1_norm"],
                "aggs": [
                    {"column": "score", "func": "sum", "as": "sum_score"},
                    {"column": "flag", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_score", "ascending": False, "nulls": "last"},
                    {"column": "k1_norm", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-multi-key-semi-join-window-aggregate",
        seed=seed,
        tables=[left, allow],
        program=program,
        metadata={
            "generator_profile": "multi_key_semi_join_window_aggregate",
            "issue_inspiration": "multi-key semi membership after null normalization feeding distinct, row-number, and aggregate",
        },
    )


def generate_string_contains_anti_join_offset_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("path", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
        ],
        [
            {"id": 0, "path": "/tmp/alpha.csv", "bucket": 1, "score": 5},
            {"id": 1, "path": "/tmp/alpha.tmp", "bucket": 1, "score": None},
            {"id": 2, "path": "beta.csv", "bucket": 2, "score": 7},
            {"id": 3, "path": "", "bucket": None, "score": -1},
            {"id": 4, "path": None, "bucket": 3, "score": 2},
            {"id": 5, "path": "space value.csv", "bucket": 2, "score": None},
        ],
    )
    blocked = TableData(
        "t_v8_blocked_paths",
        [
            ColumnSpec("has_csv", "bool", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"has_csv": True, "bucket_norm": 1},
            {"has_csv": False, "bucket_norm": -1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-string-contains-anti-join-offset",
        seed,
        [
            {"op": "mutate", "column": "path_clean", "expr": {"kind": "string_strip", "source": "path"}},
            {"op": "mutate", "column": "path_nonempty", "expr": {"kind": "string_null_if_empty", "source": "path_clean"}},
            {"op": "mutate", "column": "has_csv", "expr": {"kind": "string_contains", "source": "path_nonempty", "needle": ".csv"}},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "anti_join",
                "table": "t_v8_blocked_paths",
                "left_on": ["has_csv", "bucket"],
                "right_on": ["has_csv", "bucket_norm"],
            },
            {
                "op": "case_when",
                "as": "score_band",
                "condition": {"column": "score", "cmp": "range_closed", "value": [0, 5]},
                "then": "mid",
                "else": "edge",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "score_band", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "path_nonempty", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": seed % 2},
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-contains-anti-join-offset",
        seed=seed,
        tables=[left, blocked],
        program=program,
        metadata={
            "generator_profile": "string_contains_anti_join_offset",
            "issue_inspiration": "nullable string contains used as a join key in anti membership before offset/top-k",
        },
    )


def generate_bool_case_distinct_groupby_union_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
        ColumnSpec("x", "int", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "grp": "a", "flag": True, "x": 1},
            {"id": 1, "grp": "a", "flag": False, "x": 2},
            {"id": 2, "grp": "a", "flag": None, "x": None},
            {"id": 3, "grp": None, "flag": True, "x": 3},
        ],
    )
    append = TableData(
        "t_v8_bool_append",
        columns,
        [
            {"id": 4, "grp": "b", "flag": False, "x": -1},
            {"id": 5, "grp": "b", "flag": None, "x": 4},
            {"id": 6, "grp": None, "flag": None, "x": None},
            {"id": 7, "grp": "a", "flag": True, "x": 1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-bool-case-distinct-groupby-union",
        seed,
        [
            {"op": "union_all", "table": "t_v8_bool_append"},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_bucket",
                "else": "false_or_null_bucket",
            },
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "distinct", "columns": ["grp", "flag_bucket", "x"]},
            {
                "op": "groupby",
                "keys": ["grp", "flag_bucket"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "x", "func": "count", "as": "count_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-case-distinct-groupby-union",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "bool_case_distinct_groupby_union",
            "issue_inspiration": "three-valued boolean case bucket through union, distinct, and grouped aggregate",
        },
    )


def generate_left_join_filter_distinct_window_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "k": "a", "score": 10, "flag": True},
            {"id": 1, "k": "a", "score": None, "flag": False},
            {"id": 2, "k": "b", "score": 5, "flag": None},
            {"id": 3, "k": None, "score": 7, "flag": True},
            {"id": 4, "k": "", "score": -1, "flag": False},
            {"id": 5, "k": "c", "score": None, "flag": None},
        ],
    )
    dim = TableData(
        "t_v9_join_dim",
        [
            ColumnSpec("k_norm", "str", nullable=False),
            ColumnSpec("weight", "int", nullable=True),
            ColumnSpec("enabled", "bool", nullable=True),
        ],
        [
            {"k_norm": "a", "weight": 2, "enabled": True},
            {"k_norm": "a", "weight": None, "enabled": None},
            {"k_norm": "b", "weight": 3, "enabled": False},
            {"k_norm": "missing", "weight": -1, "enabled": True},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-left-join-filter-distinct-window",
        seed,
        [
            {"op": "mutate", "column": "k_nonempty", "expr": {"kind": "string_null_if_empty", "source": "k"}},
            {"op": "coalesce", "columns": ["k_nonempty", "k"], "as": "k_norm", "fallback": "missing"},
            {"op": "join", "table": "t_v9_join_dim", "left_on": "k_norm", "right_on": "k_norm", "how": "left"},
            {"op": "filter", "column": "enabled", "cmp": "bool_is_true", "value": None},
            {"op": "fill_null", "column": "score", "value": 0},
            {"op": "fill_null", "column": "weight", "value": 0},
            {"op": "distinct", "columns": ["k_norm", "score", "weight", "flag"]},
            {
                "op": "running_sum",
                "source": "score",
                "column": "run_score",
                "partition_by": ["k_norm"],
                "order_by": [
                    {"column": "weight", "ascending": False, "nulls": "last"},
                    {"column": "score", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["k_norm"],
                "order_by": [
                    {"column": "run_score", "ascending": False, "nulls": "last"},
                    {"column": "flag", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "k_norm", "ascending": True, "nulls": "last"},
                    {"column": "run_score", "ascending": False, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-left-join-filter-distinct-window",
        seed=seed,
        tables=[left, dim],
        program=program,
        metadata={
            "generator_profile": "left_join_filter_distinct_window",
            "issue_inspiration": "left join null extension filtered by nullable boolean before distinct and window ranking",
        },
    )


def generate_cast_groupby_membership_case(seed: int) -> Case:
    rows = [
        {"id": 0, "grp": "a", "num_s": "1", "amount_s": "10", "flag": True},
        {"id": 1, "grp": "a", "num_s": "01", "amount_s": None, "flag": False},
        {"id": 2, "grp": "b", "num_s": "-2", "amount_s": "-5", "flag": None},
        {"id": 3, "grp": "b", "num_s": "", "amount_s": "0", "flag": True},
        {"id": 4, "grp": None, "num_s": None, "amount_s": "7", "flag": False},
        {"id": 5, "grp": "c", "num_s": "9007199254740993", "amount_s": "3", "flag": None},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("num_s", "str", nullable=True),
            ColumnSpec("amount_s", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    allowed = TableData(
        "t_v9_cast_allowed",
        [ColumnSpec("num_value", "int", nullable=False)],
        [{"num_value": 1}, {"num_value": -2}, {"num_value": 9007199254740993}],
    )
    program = Program(
        f"prog-{seed:08d}-cast-groupby-membership",
        seed,
        [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {
                "op": "mutate",
                "column": "amount_value",
                "expr": {"kind": "cast", "source": "amount_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "table": "t_v9_cast_allowed", "left_on": "num_value", "right_on": "num_value"},
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "amount_value", "value": 0},
            {
                "op": "case_when",
                "as": "amount_band",
                "condition": {"column": "amount_value", "cmp": "range_closed", "value": [-5, 5]},
                "then": "small",
                "else": "large",
            },
            {
                "op": "groupby",
                "keys": ["grp", "amount_band"],
                "aggs": [
                    {"column": "amount_value", "func": "sum", "as": "sum_amount"},
                    {"column": "flag", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_amount", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-cast-groupby-membership",
        seed=seed,
        tables=[table, allowed],
        program=program,
        metadata={
            "generator_profile": "cast_groupby_membership",
            "issue_inspiration": "integer-string casts with empty/null text feeding membership and grouped aggregate",
        },
    )


def generate_null_sort_window_union_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("score", "int", nullable=True),
        ColumnSpec("delta", "int", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "grp": "a", "score": None, "delta": 1},
            {"id": 1, "grp": "a", "score": 10, "delta": None},
            {"id": 2, "grp": None, "score": 10, "delta": -1},
            {"id": 3, "grp": "b", "score": 0, "delta": 2},
        ],
    )
    append = TableData(
        "t_v9_null_sort_append",
        columns,
        [
            {"id": 4, "grp": "b", "score": None, "delta": None},
            {"id": 5, "grp": None, "score": -5, "delta": 3},
            {"id": 6, "grp": "c", "score": 10, "delta": -2},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-null-sort-window-union",
        seed,
        [
            {"op": "union_all", "table": "t_v9_null_sort_append"},
            {"op": "fill_null", "column": "delta", "value": 0},
            {
                "op": "sort",
                "keys": [
                    {"column": "score", "ascending": False, "nulls": "first"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": seed % 3},
            {
                "op": "running_sum",
                "source": "delta",
                "column": "run_delta",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "first"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "run_delta", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "first"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "run_delta", "func": "max", "as": "max_run_delta"},
                    {"column": "score", "func": "count", "as": "score_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_run_delta", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-sort-window-union",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "null_sort_window_union",
            "issue_inspiration": "explicit nulls-first ordering across union, offset, window, and aggregate",
        },
    )


def generate_date_part_membership_distinct_join_case(seed: int) -> Case:
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("dt", "str", nullable=True),
            ColumnSpec("acct", "str", nullable=True),
            ColumnSpec("amount", "int", nullable=True),
        ],
        [
            {"id": 0, "dt": "2024-01-31", "acct": "a", "amount": 5},
            {"id": 1, "dt": "2024-02-29", "acct": "a", "amount": None},
            {"id": 2, "dt": "2023-12-31", "acct": "b", "amount": -1},
            {"id": 3, "dt": "", "acct": "b", "amount": 7},
            {"id": 4, "dt": None, "acct": None, "amount": 0},
        ],
    )
    allowed = TableData(
        "t_v9_year_allow",
        [
            ColumnSpec("year", "int", nullable=False),
            ColumnSpec("month", "int", nullable=False),
        ],
        [{"year": 2024, "month": 1}, {"year": 2024, "month": 2}, {"year": 2023, "month": 12}],
    )
    dim = TableData(
        "t_v9_acct_dim",
        [
            ColumnSpec("acct_norm", "str", nullable=False),
            ColumnSpec("tier", "str", nullable=True),
        ],
        [
            {"acct_norm": "a", "tier": "gold"},
            {"acct_norm": "b", "tier": "silver"},
            {"acct_norm": "missing", "tier": None},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-date-part-membership-distinct-join",
        seed,
        [
            {"op": "mutate", "column": "year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "filter", "column": "year", "cmp": "is_not_null", "value": None},
            {
                "op": "semi_join",
                "table": "t_v9_year_allow",
                "left_on": ["year", "month"],
                "right_on": ["year", "month"],
            },
            {"op": "coalesce", "columns": ["acct", "dt"], "as": "acct_norm", "fallback": "missing"},
            {"op": "distinct", "columns": ["acct_norm", "year", "month", "amount"]},
            {"op": "join", "table": "t_v9_acct_dim", "left_on": "acct_norm", "right_on": "acct_norm", "how": "left"},
            {"op": "fill_null", "column": "amount", "value": 0},
            {
                "op": "groupby",
                "keys": ["tier", "year"],
                "aggs": [{"column": "amount", "func": "sum", "as": "sum_amount"}],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "year", "ascending": True, "nulls": "last"},
                    {"column": "tier", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-date-part-membership-distinct-join",
        seed=seed,
        tables=[fact, allowed, dim],
        program=program,
        metadata={
            "generator_profile": "date_part_membership_distinct_join",
            "issue_inspiration": "date-part extraction on nullable date strings feeding multi-key membership, distinct, and left join",
        },
    )


def generate_coalesce_case_anti_join_aggregate_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k1", "str", nullable=True),
            ColumnSpec("k2", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "k1": "a", "k2": None, "bucket": 1, "x": 5},
            {"id": 1, "k1": None, "k2": "a", "bucket": 1, "x": None},
            {"id": 2, "k1": "", "k2": "empty", "bucket": None, "x": -1},
            {"id": 3, "k1": "b", "k2": None, "bucket": 2, "x": 7},
            {"id": 4, "k1": None, "k2": None, "bucket": None, "x": 0},
        ],
    )
    blocked = TableData(
        "t_v9_blocked_keys",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
            ColumnSpec("band", "str", nullable=False),
        ],
        [
            {"key_norm": "a", "bucket_norm": 1, "band": "non_negative"},
            {"key_norm": "missing", "bucket_norm": -1, "band": "non_negative"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-coalesce-case-anti-join-aggregate",
        seed,
        [
            {"op": "mutate", "column": "k1_nonempty", "expr": {"kind": "string_null_if_empty", "source": "k1"}},
            {"op": "coalesce", "columns": ["k1_nonempty", "k2"], "as": "key_norm", "fallback": "missing"},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "case_when",
                "as": "band",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "non_negative",
                "else": "negative",
            },
            {
                "op": "anti_join",
                "table": "t_v9_blocked_keys",
                "left_on": ["key_norm", "bucket", "band"],
                "right_on": ["key_norm", "bucket_norm", "band"],
            },
            {"op": "distinct", "columns": ["key_norm", "bucket", "band", "x"]},
            {
                "op": "groupby",
                "keys": ["key_norm", "band"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "bucket", "func": "count", "as": "bucket_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "key_norm", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-coalesce-case-anti-join-aggregate",
        seed=seed,
        tables=[left, blocked],
        program=program,
        metadata={
            "generator_profile": "coalesce_case_anti_join_aggregate",
            "issue_inspiration": "coalesced nullable string keys and case buckets used as multi-key anti join before aggregate",
        },
    )


def generate_string_token_transform_join_window_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("text", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
        ],
        [
            {"id": 0, "text": "alpha beta", "bucket": 1, "score": 5},
            {"id": 1, "text": "Alpha-beta", "bucket": 1, "score": None},
            {"id": 2, "text": "beta gamma", "bucket": 2, "score": 7},
            {"id": 3, "text": "", "bucket": None, "score": -1},
            {"id": 4, "text": None, "bucket": 3, "score": 2},
            {"id": 5, "text": "space value", "bucket": 2, "score": None},
        ],
    )
    allow = TableData(
        "t_v10_token_allow",
        [
            ColumnSpec("token_key", "str", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"token_key": "alpha", "bucket_norm": 1},
            {"token_key": "Alpha-beta", "bucket_norm": 1},
            {"token_key": "beta", "bucket_norm": 2},
            {"token_key": "space", "bucket_norm": 2},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-string-token-transform-join-window",
        seed,
        [
            {"op": "mutate", "column": "text_clean", "expr": {"kind": "string_strip", "source": "text"}},
            {
                "op": "mutate",
                "column": "text_us",
                "expr": {"kind": "string_replace", "source": "text_clean", "old": " ", "new": "_"},
            },
            {"op": "mutate", "column": "prefix", "expr": {"kind": "string_slice", "source": "text_us", "start": 0, "length": 5}},
            {
                "op": "mutate",
                "column": "token",
                "expr": {"kind": "string_split_part", "source": "text_us", "sep": "_", "index": 0},
            },
            {"op": "mutate", "column": "label", "expr": {"kind": "string_concat", "source": "token", "other": "prefix", "sep": "-"}},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {
                "op": "semi_join",
                "table": "t_v10_token_allow",
                "left_on": ["token", "bucket"],
                "right_on": ["token_key", "bucket_norm"],
            },
            {"op": "fill_null", "column": "score", "value": 0},
            {"op": "distinct", "columns": ["label", "token", "bucket", "score"]},
            {
                "op": "running_sum",
                "source": "score",
                "column": "run_score",
                "partition_by": ["label"],
                "order_by": [
                    {"column": "token", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "run_score", "ascending": False, "nulls": "last"},
                    {"column": "label", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-token-transform-join-window",
        seed=seed,
        tables=[left, allow],
        program=program,
        metadata={
            "generator_profile": "string_token_transform_join_window",
            "issue_inspiration": "string replace/slice/split/concat keys feeding membership, distinct, and window ordering",
        },
    )


def generate_prefix_suffix_bool_membership_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("path", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "path": "/tmp/a.csv", "bucket": 1, "x": 5},
            {"id": 1, "path": "/tmp/a.tmp", "bucket": 1, "x": None},
            {"id": 2, "path": "data/b.csv", "bucket": 2, "x": 7},
            {"id": 3, "path": "", "bucket": None, "x": -1},
            {"id": 4, "path": None, "bucket": 3, "x": 2},
            {"id": 5, "path": "/tmp/space value.csv", "bucket": 2, "x": None},
        ],
    )
    blocked = TableData(
        "t_v10_prefix_suffix_blocked",
        [
            ColumnSpec("starts_tmp", "bool", nullable=False),
            ColumnSpec("ends_csv", "bool", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"starts_tmp": True, "ends_csv": True, "bucket_norm": 1},
            {"starts_tmp": False, "ends_csv": False, "bucket_norm": -1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-prefix-suffix-bool-membership",
        seed,
        [
            {"op": "mutate", "column": "path_clean", "expr": {"kind": "string_strip", "source": "path"}},
            {"op": "mutate", "column": "starts_tmp", "expr": {"kind": "string_starts_with", "source": "path_clean", "needle": "/tmp"}},
            {"op": "mutate", "column": "ends_csv", "expr": {"kind": "string_ends_with", "source": "path_clean", "needle": ".csv"}},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "anti_join",
                "table": "t_v10_prefix_suffix_blocked",
                "left_on": ["starts_tmp", "ends_csv", "bucket"],
                "right_on": ["starts_tmp", "ends_csv", "bucket_norm"],
            },
            {
                "op": "case_when",
                "as": "x_band",
                "condition": {"column": "x", "cmp": "range_closed", "value": [0, 5]},
                "then": "mid",
                "else": "edge",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["starts_tmp", "ends_csv"],
                "order_by": [
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "bucket", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["starts_tmp", "ends_csv", "x_band"],
                "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "x_band", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-prefix-suffix-bool-membership",
        seed=seed,
        tables=[left, blocked],
        program=program,
        metadata={
            "generator_profile": "prefix_suffix_bool_membership",
            "issue_inspiration": "nullable string prefix/suffix predicates used as multi-key anti membership and grouping keys",
        },
    )


def generate_numeric_clip_division_anti_window_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "int", nullable=True),
        ],
        [
            {"id": 0, "grp": "a", "x": -5, "y": 10},
            {"id": 1, "grp": "a", "x": None, "y": 3},
            {"id": 2, "grp": "b", "x": 0, "y": None},
            {"id": 3, "grp": "b", "x": 8, "y": -3},
            {"id": 4, "grp": None, "x": 2, "y": 5},
            {"id": 5, "grp": "c", "x": 5, "y": 0},
        ],
    )
    blocked = TableData(
        "t_v10_numeric_blocked",
        [
            ColumnSpec("grp", "str", nullable=False),
            ColumnSpec("x_clip", "int", nullable=False),
            ColumnSpec("band", "str", nullable=False),
        ],
        [
            {"grp": "a", "x_clip": -2, "band": "low"},
            {"grp": "missing", "x_clip": 2, "band": "mid"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-numeric-clip-division-anti-window",
        seed,
        [
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "fill_null", "column": "y", "value": 0},
            {"op": "mutate", "column": "x_clip", "expr": {"kind": "clip", "source": "x", "lower": -2, "upper": 5}},
            {"op": "mutate", "column": "y_div", "expr": {"kind": "arith_const", "source": "y", "op": "div", "value": 2}},
            {
                "op": "case_when",
                "as": "band",
                "condition": {"column": "x_clip", "cmp": "range_closed", "value": [0, 3]},
                "then": "mid",
                "else": "low",
            },
            {
                "op": "anti_join",
                "table": "t_v10_numeric_blocked",
                "left_on": ["grp", "x_clip", "band"],
                "right_on": ["grp", "x_clip", "band"],
            },
            {
                "op": "running_sum",
                "source": "x_clip",
                "column": "run_x",
                "partition_by": ["grp", "band"],
                "order_by": [
                    {"column": "y_div", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "run_x", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-numeric-clip-division-anti-window",
        seed=seed,
        tables=[left, blocked],
        program=program,
        metadata={
            "generator_profile": "numeric_clip_division_anti_window",
            "issue_inspiration": "clip and division-derived ordering across multi-key anti join and running window",
        },
    )


def generate_bool_not_union_distinct_aggregate_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("grp", "str", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
        ColumnSpec("x", "int", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "grp": "a", "flag": True, "x": 1},
            {"id": 1, "grp": "a", "flag": False, "x": None},
            {"id": 2, "grp": None, "flag": None, "x": 2},
            {"id": 3, "grp": "b", "flag": True, "x": -1},
        ],
    )
    append = TableData(
        "t_v10_bool_append",
        columns,
        [
            {"id": 4, "grp": "b", "flag": False, "x": 4},
            {"id": 5, "grp": None, "flag": True, "x": None},
            {"id": 6, "grp": "c", "flag": None, "x": 0},
            {"id": 7, "grp": "a", "flag": True, "x": 1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-bool-not-union-distinct-aggregate",
        seed,
        [
            {"op": "union_all", "table": "t_v10_bool_append"},
            {"op": "fill_null", "column": "flag", "value": False},
            {"op": "mutate", "column": "flag_inv", "expr": {"kind": "bool_not", "source": "flag"}},
            {
                "op": "case_when",
                "as": "flag_band",
                "condition": {"column": "flag_inv", "cmp": "bool_is_true", "value": None},
                "then": "inverted_true",
                "else": "inverted_false",
            },
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "distinct", "columns": ["grp", "flag_inv", "flag_band", "x"]},
            {
                "op": "row_number_filter",
                "partition_by": ["grp", "flag_band"],
                "order_by": [
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "flag_inv", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["grp", "flag_band"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag_inv", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-not-union-distinct-aggregate",
        seed=seed,
        tables=[base, append],
        program=program,
        metadata={
            "generator_profile": "bool_not_union_distinct_aggregate",
            "issue_inspiration": "nullable boolean fill/not/case through union, distinct, row-number, and aggregate",
        },
    )


def generate_outer_join_coalesce_distinct_topk_case(seed: int) -> Case:
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw_key", "str", nullable=True),
            ColumnSpec("amount", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 0, "raw_key": " Alpha ", "amount": 10, "flag": True},
            {"id": 1, "raw_key": "alpha", "amount": None, "flag": False},
            {"id": 2, "raw_key": "", "amount": -3, "flag": None},
            {"id": 3, "raw_key": None, "amount": 5, "flag": True},
            {"id": 4, "raw_key": "BETA", "amount": 0, "flag": False},
            {"id": 5, "raw_key": "space value", "amount": None, "flag": None},
        ],
    )
    dim = TableData(
        "t_v11_outer_dim",
        [
            ColumnSpec("key_norm", "str", nullable=False),
            ColumnSpec("tier", "str", nullable=True),
            ColumnSpec("weight", "int", nullable=True),
        ],
        [
            {"key_norm": "alpha", "tier": "gold", "weight": 2},
            {"key_norm": "alpha", "tier": "gold", "weight": None},
            {"key_norm": "beta", "tier": "silver", "weight": 3},
            {"key_norm": "missing", "tier": None, "weight": -1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-outer-join-coalesce-distinct-topk",
        seed,
        [
            {"op": "mutate", "column": "key_clean", "expr": {"kind": "string_strip", "source": "raw_key"}},
            {"op": "mutate", "column": "key_lower", "expr": {"kind": "string_lower", "source": "key_clean"}},
            {"op": "mutate", "column": "key_nonempty", "expr": {"kind": "string_null_if_empty", "source": "key_lower"}},
            {"op": "coalesce", "columns": ["key_nonempty", "raw_key"], "as": "key_norm", "fallback": "missing"},
            {"op": "join", "table": "t_v11_outer_dim", "left_on": "key_norm", "right_on": "key_norm", "how": "left"},
            {"op": "fill_null", "column": "amount", "value": 0},
            {"op": "fill_null", "column": "weight", "value": 0},
            {
                "op": "case_when",
                "as": "amount_band",
                "condition": {"column": "amount", "cmp": "range_closed", "value": [0, 10]},
                "then": "in_band",
                "else": "out_band",
            },
            {"op": "distinct", "columns": ["key_norm", "tier", "amount_band", "amount", "weight", "flag"]},
            {
                "op": "groupby",
                "keys": ["tier", "amount_band"],
                "aggs": [
                    {"column": "amount", "func": "sum", "as": "sum_amount"},
                    {"column": "weight", "func": "max", "as": "max_weight"},
                    {"column": "flag", "func": "count", "as": "flag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_amount", "ascending": False, "nulls": "last"},
                    {"column": "max_weight", "ascending": False, "nulls": "last"},
                    {"column": "tier", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-outer-join-coalesce-distinct-topk",
        seed=seed,
        tables=[fact, dim],
        program=program,
        metadata={
            "generator_profile": "outer_join_coalesce_distinct_topk",
            "issue_inspiration": "duplicate left join expansion with cleaned nullable keys before distinct grouped top-k",
        },
    )


def generate_chained_string_cleanup_membership_window_case(seed: int) -> Case:
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("label", "str", nullable=True),
            ColumnSpec("bucket", "int", nullable=True),
            ColumnSpec("score", "int", nullable=True),
        ],
        [
            {"id": 0, "label": " A/one.csv ", "bucket": 1, "score": 5},
            {"id": 1, "label": "a/two.tmp", "bucket": 1, "score": None},
            {"id": 2, "label": "B/one.csv", "bucket": 2, "score": 7},
            {"id": 3, "label": "", "bucket": None, "score": -1},
            {"id": 4, "label": None, "bucket": 3, "score": 2},
            {"id": 5, "label": "space value.csv", "bucket": 2, "score": None},
        ],
    )
    allow = TableData(
        "t_v11_string_allow",
        [
            ColumnSpec("prefix_key", "str", nullable=False),
            ColumnSpec("is_csv", "bool", nullable=False),
            ColumnSpec("bucket_norm", "int", nullable=False),
        ],
        [
            {"prefix_key": "a", "is_csv": True, "bucket_norm": 1},
            {"prefix_key": "b", "is_csv": True, "bucket_norm": 2},
            {"prefix_key": "space value.csv", "is_csv": True, "bucket_norm": 2},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-chained-string-cleanup-membership-window",
        seed,
        [
            {"op": "mutate", "column": "label_clean", "expr": {"kind": "string_strip", "source": "label"}},
            {"op": "mutate", "column": "label_lower", "expr": {"kind": "string_lower", "source": "label_clean"}},
            {
                "op": "mutate",
                "column": "label_slash",
                "expr": {"kind": "string_replace", "source": "label_lower", "old": "\\", "new": "/"},
            },
            {
                "op": "mutate",
                "column": "prefix_key",
                "expr": {"kind": "string_split_part", "source": "label_slash", "sep": "/", "index": 0},
            },
            {"op": "mutate", "column": "is_csv", "expr": {"kind": "string_ends_with", "source": "label_slash", "needle": ".csv"}},
            {"op": "fill_null", "column": "bucket", "value": -1},
            {"op": "fill_null", "column": "score", "value": 0},
            {
                "op": "case_when",
                "as": "csv_band",
                "condition": {"column": "is_csv", "cmp": "bool_is_true", "value": None},
                "then": "csv",
                "else": "non_csv_or_missing",
            },
            {
                "op": "semi_join",
                "table": "t_v11_string_allow",
                "left_on": ["prefix_key", "is_csv", "bucket"],
                "right_on": ["prefix_key", "is_csv", "bucket_norm"],
            },
            {
                "op": "row_number_filter",
                "partition_by": ["prefix_key", "csv_band"],
                "order_by": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "bucket", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "prefix_key", "ascending": True, "nulls": "last"},
                    {"column": "score", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": seed % 2},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-chained-string-cleanup-membership-window",
        seed=seed,
        tables=[fact, allow],
        program=program,
        metadata={
            "generator_profile": "chained_string_cleanup_membership_window",
            "issue_inspiration": "chained string cleanup and suffix predicate as multi-key membership before top-n window",
        },
    )


def generate_cast_date_union_anti_running_case(seed: int) -> Case:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("acct", "str", nullable=True),
        ColumnSpec("dt", "str", nullable=True),
        ColumnSpec("amount_s", "str", nullable=True),
    ]
    base = TableData(
        "t0",
        columns,
        [
            {"id": 0, "acct": "a", "dt": "2024-01-31", "amount_s": "10"},
            {"id": 1, "acct": "a", "dt": "2024-02-29", "amount_s": None},
            {"id": 2, "acct": "b", "dt": None, "amount_s": "-5"},
            {"id": 3, "acct": None, "dt": None, "amount_s": "0"},
        ],
    )
    append = TableData(
        "t_v11_cast_date_append",
        columns,
        [
            {"id": 4, "acct": "b", "dt": "2023-12-31", "amount_s": "7"},
            {"id": 5, "acct": None, "dt": "2024-01-01", "amount_s": "3"},
            {"id": 6, "acct": "c", "dt": "2024-03-01", "amount_s": None},
        ],
    )
    blocked = TableData(
        "t_v11_cast_date_blocked",
        [
            ColumnSpec("acct_norm", "str", nullable=False),
            ColumnSpec("year", "int", nullable=False),
            ColumnSpec("band", "str", nullable=False),
        ],
        [
            {"acct_norm": "a", "year": 2024, "band": "non_negative"},
            {"acct_norm": "missing", "year": 2024, "band": "non_negative"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-cast-date-union-anti-running",
        seed,
        [
            {"op": "union_all", "table": "t_v11_cast_date_append"},
            {
                "op": "mutate",
                "column": "amount",
                "expr": {"kind": "cast", "source": "amount_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "mutate", "column": "year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "fill_null", "column": "acct", "value": "missing"},
            {"op": "fill_null", "column": "amount", "value": 0},
            {"op": "fill_null", "column": "year", "value": -1},
            {
                "op": "case_when",
                "as": "band",
                "condition": {"column": "amount", "cmp": ">=", "value": 0},
                "then": "non_negative",
                "else": "negative",
            },
            {
                "op": "anti_join",
                "table": "t_v11_cast_date_blocked",
                "left_on": ["acct", "year", "band"],
                "right_on": ["acct_norm", "year", "band"],
            },
            {
                "op": "running_sum",
                "source": "amount",
                "column": "run_amount",
                "partition_by": ["acct", "band"],
                "order_by": [
                    {"column": "year", "ascending": True, "nulls": "last"},
                    {"column": "month", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "run_amount", "ascending": False, "nulls": "last"},
                    {"column": "acct", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-cast-date-union-anti-running",
        seed=seed,
        tables=[base, append, blocked],
        program=program,
        metadata={
            "generator_profile": "cast_date_union_anti_running",
            "issue_inspiration": "nullable integer-string cast and date-part extraction through union, anti join, and running sum",
        },
    )


def generate_post_groupby_filter_membership_topk_case(seed: int) -> Case:
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "grp": "a", "flag": True, "x": 5},
            {"id": 1, "grp": "a", "flag": None, "x": None},
            {"id": 2, "grp": "b", "flag": False, "x": -2},
            {"id": 3, "grp": "b", "flag": True, "x": 3},
            {"id": 4, "grp": None, "flag": None, "x": 0},
            {"id": 5, "grp": "c", "flag": False, "x": None},
        ],
    )
    allow = TableData(
        "t_v11_post_group_allow",
        [
            ColumnSpec("activity_band", "str", nullable=False),
            ColumnSpec("flag_seen_count", "int", nullable=False),
        ],
        [
            {"activity_band": "active", "flag_seen_count": 2},
            {"activity_band": "sparse", "flag_seen_count": 1},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-post-groupby-filter-membership-topk",
        seed,
        [
            {"op": "fill_null", "column": "grp", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "flag", "func": "count", "as": "flag_seen_count"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {"op": "filter", "column": "sum_x", "cmp": "range_closed", "value": [-2, 8]},
            {
                "op": "case_when",
                "as": "activity_band",
                "condition": {"column": "flag_seen_count", "cmp": ">=", "value": 2},
                "then": "active",
                "else": "sparse",
            },
            {
                "op": "semi_join",
                "table": "t_v11_post_group_allow",
                "left_on": ["activity_band", "flag_seen_count"],
                "right_on": ["activity_band", "flag_seen_count"],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "any_flag", "ascending": False, "nulls": "last"},
                    {"column": "grp", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-post-groupby-filter-membership-topk",
        seed=seed,
        tables=[fact, allow],
        program=program,
        metadata={
            "generator_profile": "post_groupby_filter_membership_topk",
            "issue_inspiration": "aggregate output filter and membership join before ordered top-k",
        },
    )


def generate_duplicate_key_left_join_window_aggregate_case(seed: int) -> Case:
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k", "str", nullable=True),
            ColumnSpec("seq", "int", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "k": "a", "seq": 1, "x": 2},
            {"id": 1, "k": "a", "seq": 1, "x": None},
            {"id": 2, "k": "b", "seq": 2, "x": 5},
            {"id": 3, "k": None, "seq": 3, "x": -1},
            {"id": 4, "k": "", "seq": 4, "x": 0},
        ],
    )
    right = TableData(
        "t_v11_dup_dim",
        [
            ColumnSpec("k_norm", "str", nullable=False),
            ColumnSpec("mult", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"k_norm": "a", "mult": 1, "tag": "first"},
            {"k_norm": "a", "mult": 2, "tag": "second"},
            {"k_norm": "b", "mult": None, "tag": "only"},
            {"k_norm": "missing", "mult": -1, "tag": None},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-duplicate-key-left-join-window-aggregate",
        seed,
        [
            {"op": "mutate", "column": "k_nonempty", "expr": {"kind": "string_null_if_empty", "source": "k"}},
            {"op": "coalesce", "columns": ["k_nonempty", "k"], "as": "k_norm", "fallback": "missing"},
            {"op": "join", "table": "t_v11_dup_dim", "left_on": "k_norm", "right_on": "k_norm", "how": "left"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "fill_null", "column": "mult", "value": 0},
            {"op": "mutate", "column": "weighted_x", "expr": {"kind": "arith_const", "source": "x", "op": "mul", "value": 2}},
            {
                "op": "running_sum",
                "source": "weighted_x",
                "column": "run_weighted_x",
                "partition_by": ["k_norm"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "mult", "ascending": False, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "row_number_filter",
                "partition_by": ["k_norm"],
                "order_by": [
                    {"column": "run_weighted_x", "ascending": False, "nulls": "last"},
                    {"column": "tag", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": 2,
            },
            {
                "op": "groupby",
                "keys": ["k_norm"],
                "aggs": [
                    {"column": "run_weighted_x", "func": "max", "as": "max_run_weighted_x"},
                    {"column": "tag", "func": "count", "as": "tag_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_run_weighted_x", "ascending": False, "nulls": "last"},
                    {"column": "k_norm", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-duplicate-key-left-join-window-aggregate",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "duplicate_key_left_join_window_aggregate",
            "issue_inspiration": "duplicate-key left join cardinality feeding deterministic running-sum and grouped aggregate",
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


def generate_pandas_arrow_bool_groupby_reduction_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_bool_groupby_reduction_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-bool-groupby-reduction-semantics",
        seed,
        [{"op": "arrow_bool_groupby_reduction_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-bool-groupby-reduction-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_bool_groupby_reduction_semantics",
            "issue_inspiration": "Arrow-backed nullable boolean groupby reductions should match pandas nullable boolean semantics",
            "expected_arrow_bool_groupby_reduction_mismatch": False,
            "expected_groupby_reductions": {
                "any_skipna_true": {"a": True, "b": False, "c": False, "d": True},
                "all_skipna_true": {"a": True, "b": False, "c": True, "d": False},
                "any_skipna_false": {"a": True, "b": None, "c": None, "d": True},
                "all_skipna_false": {"a": None, "b": False, "c": None, "d": False},
            },
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


def generate_bitmap_boundary_bool_aggregate_case(seed: int) -> Case:
    """Exercise nullable boolean kernels on both sides of 64-bit bitmap words."""

    row_counts = (63, 64, 65, 127, 128, 129)
    row_count = row_counts[seed % len(row_counts)]
    rows = []
    for idx in range(row_count):
        flag = None if idx % 11 == 0 else ((idx * 5 + seed) % 7 < 3)
        rows.append(
            {
                "row_id": idx,
                "grp": f"g{idx % 5}",
                "flag": flag,
                "value": (idx % 17) - 8,
            }
        )
    random.Random(seed * 65537 + 17).shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=False),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("value", "int", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-bitmap-boundary-bool-aggregate",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["grp"],
                "aggs": [
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                    {"column": "flag", "func": "count", "as": "seen_flag"},
                    {"column": "value", "func": "sum", "as": "sum_value"},
                    {"column": "row_id", "func": "count", "as": "row_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [{"column": "grp", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bitmap-boundary-bool-aggregate",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bitmap_boundary_bool_aggregate",
            "exploration_axis": "nullable_bitmap_word_boundary",
            "row_count": row_count,
            "boundary_values": list(row_counts),
        },
    )


def generate_vector_boundary_groupby_distinct_case(seed: int) -> Case:
    """Cross common 1K/2K execution-batch boundaries before hash aggregation."""

    row_counts = (1023, 1024, 1025, 2047, 2048, 2049)
    row_count = row_counts[seed % len(row_counts)]
    rows = []
    for idx in range(row_count):
        logical = idx % 389
        rows.append(
            {
                "bucket": logical % 23,
                "subkey": (logical * 7) % 31,
                "marker": f"m{logical % 13}",
                "value": (logical * 17) % 101 - 50,
            }
        )
    table = TableData(
        "t0",
        [
            ColumnSpec("bucket", "int", nullable=False),
            ColumnSpec("subkey", "int", nullable=False),
            ColumnSpec("marker", "str", nullable=False),
            ColumnSpec("value", "int", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-vector-boundary-groupby-distinct",
        seed,
        [
            {"op": "distinct", "columns": ["bucket", "subkey", "marker", "value"]},
            {
                "op": "groupby",
                "keys": ["bucket"],
                "aggs": [
                    {"column": "marker", "func": "count", "as": "row_count"},
                    {"column": "subkey", "func": "nunique", "as": "subkey_count"},
                    {"column": "value", "func": "sum", "as": "sum_value"},
                    {"column": "value", "func": "min", "as": "min_value"},
                    {"column": "value", "func": "max", "as": "max_value"},
                ],
            },
            {
                "op": "sort",
                "keys": [{"column": "bucket", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-vector-boundary-groupby-distinct",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "vector_boundary_groupby_distinct",
            "exploration_axis": "execution_vector_boundary",
            "row_count": row_count,
            "boundary_values": list(row_counts),
        },
    )


def generate_wide_schema_projection_boundary_case(seed: int) -> Case:
    """Select non-adjacent columns around common 32/64-column mask boundaries."""

    payload_counts = (31, 32, 33, 63, 64, 65)
    payload_count = payload_counts[seed % len(payload_counts)]
    columns = [ColumnSpec("row_id", "int", nullable=False)] + [
        ColumnSpec(f"p_{idx}", "int", nullable=idx % 7 == 0)
        for idx in range(payload_count)
    ]
    rows = []
    for row_id in range(19):
        row: dict[str, Any] = {"row_id": row_id}
        for idx in range(payload_count):
            row[f"p_{idx}"] = None if idx % 7 == 0 and (row_id + idx) % 5 == 0 else row_id * 100 + idx
        rows.append(row)
    middle = payload_count // 2
    last = payload_count - 1
    table = TableData("t0", columns, rows)
    selected_columns = ["row_id", "p_0", f"p_{middle}", f"p_{last}", "edge_shift"]
    program = Program(
        f"prog-{seed:08d}-wide-schema-projection-boundary",
        seed,
        [
            {"op": "fill_null", "column": "p_0", "value": -1},
            {"op": "fill_null", "column": f"p_{middle}", "value": -1},
            {"op": "fill_null", "column": f"p_{last}", "value": -1},
            {
                "op": "mutate",
                "column": "edge_shift",
                "expr": {"kind": "add_const", "source": f"p_{last}", "value": 1},
            },
            {"op": "filter", "column": f"p_{middle}", "cmp": ">=", "value": -1},
            {"op": "select", "columns": selected_columns},
            {"op": "distinct", "columns": selected_columns},
            {
                "op": "sort",
                "keys": [{"column": "row_id", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-wide-schema-projection-boundary",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "wide_schema_projection_boundary",
            "exploration_axis": "wide_schema_projection_mask_boundary",
            "payload_column_count": payload_count,
            "boundary_values": list(payload_counts),
        },
    )


def generate_skewed_join_multiplicity_case(seed: int) -> Case:
    """Stress hash joins with a hot key while retaining exact bag semantics."""

    row_counts = (127, 128, 129, 255, 256, 257)
    row_count = row_counts[seed % len(row_counts)]
    left_rows = []
    for row_id in range(row_count):
        if row_id < (row_count * 3) // 4:
            join_key = 0
        else:
            join_key = 1 + (row_id % 11)
        left_rows.append(
            {
                "row_id": row_id,
                "join_key": join_key,
                "measure": row_id % 19 - 9,
            }
        )
    right_rows = []
    for join_key in range(12):
        duplicate_count = 5 if join_key == 0 else 1 + ((join_key + seed) % 3)
        for duplicate_id in range(duplicate_count):
            right_rows.append(
                {
                    "dim_key": join_key,
                    "dim_id": join_key * 10 + duplicate_id,
                    "weight": duplicate_id + 1,
                }
            )
    left = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("join_key", "int", nullable=False),
            ColumnSpec("measure", "int", nullable=False),
        ],
        left_rows,
    )
    right = TableData(
        "t_skew_dim",
        [
            ColumnSpec("dim_key", "int", nullable=False),
            ColumnSpec("dim_id", "int", nullable=False),
            ColumnSpec("weight", "int", nullable=False),
        ],
        right_rows,
    )
    program = Program(
        f"prog-{seed:08d}-skewed-join-multiplicity",
        seed,
        [
            {
                "op": "join",
                "table": "t_skew_dim",
                "left_on": "join_key",
                "right_on": "dim_key",
                "how": "inner",
            },
            {
                "op": "groupby",
                "keys": ["join_key"],
                "aggs": [
                    {"column": "row_id", "func": "count", "as": "expanded_rows"},
                    {"column": "dim_id", "func": "nunique", "as": "matched_dim_rows"},
                    {"column": "measure", "func": "sum", "as": "sum_measure"},
                    {"column": "weight", "func": "sum", "as": "sum_weight"},
                ],
            },
            {
                "op": "sort",
                "keys": [{"column": "join_key", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-skewed-join-multiplicity",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "skewed_join_multiplicity",
            "exploration_axis": "hash_join_hot_key_multiplicity",
            "row_count": row_count,
            "hot_key_rows": (row_count * 3) // 4,
        },
    )


def generate_utf8_slice_length_groupby_case(seed: int) -> Case:
    """Compare code-point length/slice behavior without relying on string collation."""

    values = [
        "a" * 31 + "é",
        "中" * 16 + "x",
        "🙂" * 8 + "tail",
        "e\u0301" * 9,
        "Straße",
        "Δelta-中文-🙂",
        "plain-ascii",
        "éclair",
    ]
    slice_lengths = (1, 2, 3, 7, 8, 9)
    slice_length = slice_lengths[seed % len(slice_lengths)]
    rows = [
        {"row_id": idx, "text_value": value}
        for idx, value in enumerate(values + list(reversed(values)))
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("text_value", "str", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-utf8-slice-length-groupby",
        seed,
        [
            {
                "op": "mutate",
                "column": "text_prefix",
                "expr": {
                    "kind": "string_slice",
                    "source": "text_value",
                    "start": 0,
                    "length": slice_length,
                },
            },
            {
                "op": "mutate",
                "column": "text_length",
                "expr": {"kind": "string_length", "source": "text_value"},
            },
            {
                "op": "groupby",
                "keys": ["text_prefix", "text_length"],
                "aggs": [
                    {"column": "row_id", "func": "count", "as": "row_count"},
                    {"column": "text_value", "func": "nunique", "as": "distinct_text_count"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-utf8-slice-length-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "utf8_slice_length_groupby",
            "exploration_axis": "utf8_codepoint_slice_length",
            "slice_length": slice_length,
            "uses_explicit_output_order": False,
        },
    )


def generate_unique_order_window_tiebreak_case(seed: int) -> Case:
    """Exercise window planning with duplicate primary keys and a unique tiebreaker."""

    row_counts = (31, 32, 33, 63, 64, 65)
    row_count = row_counts[seed % len(row_counts)]
    rows = [
        {
            "row_id": row_id,
            "grp": f"g{row_id % 4}",
            "score": (row_id * 7) % 9,
            "amount": row_id % 13 - 6,
        }
        for row_id in range(row_count)
    ]
    random.Random(seed * 131071 + 29).shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=False),
            ColumnSpec("score", "int", nullable=False),
            ColumnSpec("amount", "int", nullable=False),
        ],
        rows,
    )
    order_keys = [
        {"column": "score", "ascending": False, "nulls": "last"},
        {"column": "row_id", "ascending": True, "nulls": "last"},
    ]
    program = Program(
        f"prog-{seed:08d}-unique-order-window-tiebreak",
        seed,
        [
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": order_keys,
                "cmp": "<=",
                "value": 4,
            },
            {
                "op": "running_sum",
                "source": "amount",
                "column": "running_amount",
                "partition_by": ["grp"],
                "order_by": order_keys,
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    *order_keys,
                ],
            },
            {"op": "select", "columns": ["grp", "score", "row_id", "running_amount"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-unique-order-window-tiebreak",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "unique_order_window_tiebreak",
            "exploration_axis": "deterministic_window_tie_break",
            "row_count": row_count,
            "unique_order_columns": ["score", "row_id"],
        },
    )
