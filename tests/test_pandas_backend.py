import warnings
from types import SimpleNamespace

import pytest

from datadiff.backends.pandas_backend import _records_without_duplicate_column_warning
from datadiff.backends.pandas_backend import PandasBackend
from datadiff.datagen import generate_case
from datadiff.dsl import ColumnSpec, Program, TableData


def test_records_without_duplicate_column_warning_suppresses_pandas_noise():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2]], columns=["x", "x"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = _records_without_duplicate_column_warning(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = df.to_dict("records")

    assert rows == expected
    assert not [
        warning
        for warning in caught
        if "DataFrame columns are not unique" in str(warning.message)
    ]


def test_pandas_arrow_bool_groupby_reduction_probe_matches_reference_semantics():
    pytest.importorskip("pyarrow")
    case = generate_case(148, profile="pandas_arrow_bool_groupby_reduction_semantics")

    result = PandasBackend().run(case.tables, case.program)

    assert result.status == "ok"
    assert result.data.to_dict("records") == [{"arrow_bool_groupby_reduction_mismatch": False}]


def test_pandas_string_slice_does_not_overwrite_backend_timer(monkeypatch):
    case = generate_case(0, profile="utf8_slice_length_groupby")
    ticks = iter((100.0, 100.25))
    monkeypatch.setattr(
        "datadiff.backends.pandas_backend.time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )

    result = PandasBackend().run(case.tables, case.program)

    assert result.status == "ok"
    assert result.duration_ms == pytest.approx(250.0)


def test_pandas_empty_case_when_preserves_string_output_type():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False), ColumnSpec("flag", "bool")],
        [{"id": 1, "flag": True}],
    )
    program = Program(
        "p-empty-case-when-string",
        1,
        [
            {"op": "filter", "column": "id", "cmp": ">", "value": 1},
            {
                "op": "case_when",
                "as": "label",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "yes",
                "else": "no",
            },
            {
                "op": "mutate",
                "column": "label_length",
                "expr": {"kind": "string_length", "source": "label"},
            },
        ],
    )

    result = PandasBackend().run([table], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert result.data.empty
    assert str(result.data["label"].dtype) == "string"


def test_pandas_all_null_float_column_preserves_type_for_abs():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False), ColumnSpec("y", "float")],
        [{"id": 0, "y": None}],
    )
    program = Program(
        "p-all-null-float-abs",
        2,
        [
            {
                "op": "mutate",
                "column": "magnitude",
                "expr": {"kind": "abs", "source": "y"},
            }
        ],
    )

    result = PandasBackend().run([table], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert str(result.data["y"].dtype) == "Float64"
    assert str(result.data["magnitude"].dtype) == "Float64"
    assert result.data["magnitude"].isna().tolist() == [True]


def test_pandas_bool_not_restores_nullable_dtype_after_row_materialization():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("x", "float"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1.0, "flag": None},
            {"id": 1, "x": 2.0, "flag": True},
            {"id": 2, "x": 3.0, "flag": False},
        ],
    )
    program = Program(
        "p-nullable-bool-not-after-running-sum",
        3,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "order_by": [
                    {"column": "id", "ascending": True, "nulls": "last"}
                ],
            },
            {
                "op": "mutate",
                "column": "not_flag",
                "expr": {"kind": "bool_not", "source": "flag"},
            },
        ],
    )

    result = PandasBackend().run([table], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert str(result.data["not_flag"].dtype) == "boolean"
    assert result.data["not_flag"].isna().tolist() == [True, False, False]
    assert result.data["not_flag"].iloc[1:].tolist() == [False, True]
