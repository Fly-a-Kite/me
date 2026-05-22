from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff import reducer


def test_reducer_preserves_extra_join_tables_when_minimizing(monkeypatch):
    seen_table_counts = []

    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        seen_table_counts.append(len(candidate.tables))
        return {"findings": [{"kind": "semantic_output_mismatch"}]}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-join",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 2}, {"id": 2, "x": 3}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("j", "int")], [{"id": 1, "j": 10}]),
        ],
        Program("prog-join", 1, [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}]),
    )

    reduced = reducer.reduce_case(case, backends=["pandas", "polars"], target_kinds=["semantic_output_mismatch"])

    assert len(reduced.tables) == 2
    assert reduced.tables[1].name == "t1"
    assert seen_table_counts
    assert all(count == 2 for count in seen_table_counts)


def test_reducer_minimizes_rows_in_referenced_join_tables(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        if len(candidate.tables) != 2:
            return {"findings": []}
        t1 = candidate.tables[1]
        column_names = {column.name for column in t1.columns}
        if column_names == {"id", "j"} and any(row.get("j") == 99 for row in t1.rows):
            return {"findings": [{"kind": "semantic_output_mismatch"}]}
        return {"findings": []}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-join-right-rows",
        7,
        [
            TableData("t0", [ColumnSpec("id", "int")], [{"id": 1}]),
            TableData(
                "t1",
                [ColumnSpec("id", "int"), ColumnSpec("j", "int")],
                [{"id": 1, "j": 1}, {"id": 1, "j": 99}, {"id": 1, "j": 100}],
            ),
        ],
        Program("prog-join-right-rows", 7, [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}]),
    )

    reduced = reducer.reduce_case(case, backends=["pandas", "polars"], target_kinds=["semantic_output_mismatch"])

    assert len(reduced.tables) == 2
    assert reduced.tables[1].rows == [{"id": 1, "j": 99}]


def test_reducer_rejects_same_kind_with_different_root_cause(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        return {"findings": [{"kind": "semantic_output_mismatch", "root_cause": "join_semantics"}]}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-filter",
        2,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-filter", 2, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )

    reduced = reducer.reduce_case(
        case,
        backends=["pandas", "polars"],
        target_kinds=["semantic_output_mismatch"],
        target_roots=["filter_predicate"],
    )

    assert reduced.to_dict() == case.to_dict()


def test_reducer_rejects_false_positive_reduction(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        return {
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "filter_predicate",
                    "triage_verdict": "generator_false_positive",
                    "false_positive": True,
                }
            ]
        }

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-filter-fp",
        3,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-filter-fp", 3, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )

    reduced = reducer.reduce_case(
        case,
        backends=["pandas", "polars"],
        target_kinds=["semantic_output_mismatch"],
        target_roots=["filter_predicate"],
    )

    assert reduced.to_dict() == case.to_dict()


def test_reducer_keeps_sort_when_later_limit_depends_on_it(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        if any(op.get("op") == "limit" for op in candidate.program.operations):
            return {"findings": [{"kind": "semantic_output_mismatch", "root_cause": "ordering_or_limit"}]}
        return {"findings": []}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-sort-limit",
        4,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-sort-limit",
            4,
            [
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    reduced = reducer.reduce_case(case, backends=["pandas", "datafusion"])

    assert reduced.program.operations == case.program.operations


def test_reducer_can_remove_superseded_sort_before_limit(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        ops = [op.get("op") for op in candidate.program.operations]
        if ops == ["sort", "limit"]:
            return {"findings": [{"kind": "semantic_output_mismatch"}]}
        return {"findings": []}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-superseded-sort",
        5,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-superseded-sort",
            5,
            [
                {"op": "sort", "columns": ["x"], "ascending": True},
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    reduced = reducer.reduce_case(case, backends=["pandas", "datafusion"])

    assert reduced.program.operations == [
        {"op": "sort", "columns": ["x"], "ascending": False},
        {"op": "limit", "n": 1},
    ]


def test_reducer_removes_unreferenced_extra_tables(monkeypatch):
    def fake_run_loaded_case(candidate, backends, config=None, save_artifact=False):
        if len(candidate.tables) == 1:
            return {"findings": [{"kind": "semantic_output_mismatch"}]}
        return {"findings": []}

    monkeypatch.setattr(reducer, "run_loaded_case", fake_run_loaded_case)
    case = Case(
        "case-extra-table",
        6,
        [
            TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program("prog-extra-table", 6, [{"op": "limit", "n": 1}]),
    )

    reduced = reducer.reduce_case(case, backends=["pandas", "datafusion"])

    assert [table.name for table in reduced.tables] == ["t0"]
