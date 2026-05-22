import json

import numpy as np
import pandas as pd

from datadiff.backends.base import BackendResult
from datadiff.dsl import Program
from datadiff.normalizer import _norm_value, normalize_result


def test_normalizer_handles_duplicate_column_names_without_series_values():
    df = pd.DataFrame([[1, 2, "a"]])
    df.columns = ["x", "x", "s"]

    result = normalize_result(BackendResult("pandas", "ok", data=df), Program("prog", 1, []))

    assert result.status == "ok"
    assert result.columns == ["s", "x", "x"]
    assert result.rows == [["a", 1, 2]]
    json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def test_normalizer_coerces_numpy_float_scalars_to_builtin_float():
    value = _norm_value(np.float64(-0.5))

    assert value == -0.5
    assert type(value) is float


def test_normalizer_collapses_near_integer_float_after_rounding():
    assert _norm_value(5.000000000000001) == 5
    assert type(_norm_value(5.000000000000001)) is int


def test_normalizer_uses_stable_json_row_order_for_mixed_null_and_float_rows():
    df = pd.DataFrame(
        [
            [None, 3, np.float64(-0.5)],
            [None, 3, None],
        ],
        columns=["a", "b", "c"],
    )

    result = normalize_result(BackendResult("pandas", "ok", data=df), Program("prog", 1, []))

    assert result.rows == [[None, 3, -0.5], [None, 3, None]]


def test_normalizer_preserves_explicit_sort_order_for_order_sensitive_programs():
    df = pd.DataFrame([[2], [1]], columns=["x"])
    program = Program("prog", 1, [{"op": "sort", "columns": ["x"], "ascending": False}])

    result = normalize_result(BackendResult("pandas", "ok", data=df), program)

    assert program.order_sensitive is True
    assert result.rows == [[2], [1]]


def test_order_sensitive_survives_limit_and_offset_after_sort():
    program = Program(
        "prog",
        1,
        [
            {"op": "sort", "columns": ["x"], "ascending": False},
            {"op": "limit", "n": 2},
            {"op": "offset", "n": 1},
        ],
    )

    assert program.order_sensitive is True
