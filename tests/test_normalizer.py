import json
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

from datadiff.backends.base import BackendResult, NativeRows
from datadiff.canonicalization import canonical_key, result_comparison_key
from datadiff.dsl import Program
from datadiff.normalizer import NormalizedResult, _norm_value, normalize_result
from datadiff.semantic_values import LOSSLESS_VALUE_SCHEMA_VERSION, encode_semantic_value


def test_normalizer_handles_duplicate_column_names_without_series_values():
    df = pd.DataFrame([[1, 2, "a"]])
    df.columns = ["x", "x", "s"]

    result = normalize_result(BackendResult("pandas", "ok", data=df), Program("prog", 1, []))

    assert result.status == "ok"
    assert result.columns == ["s", "x", "x"]
    assert result.rows == [["a", 1, 2]]
    json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def test_normalizer_preserves_native_rows_column_types():
    data = NativeRows(
        columns=["x", "label"],
        row_values=[[1, "one"]],
        column_types=["int64", "string"],
    )

    result = normalize_result(
        BackendResult("native", "ok", data=data),
        Program("prog", 1, []),
    )

    assert result.column_types == ["string", "int64"]
    assert result.logical_column_types == ["string", "integer:64:signed"]


def test_normalizer_coerces_numpy_float_scalars_to_builtin_float():
    value = _norm_value(np.float64(-0.5))

    assert value == -0.5
    assert type(value) is float


def test_lossless_semantic_values_distinguish_null_nan_and_signed_zero():
    null_value = encode_semantic_value(None)
    nan_value = encode_semantic_value(float("nan"))
    positive_zero = encode_semantic_value(0.0)
    negative_zero = encode_semantic_value(-0.0)

    assert null_value["kind"] == "null"
    assert nan_value["kind"] == "nan"
    assert null_value != nan_value
    assert positive_zero["kind"] == "float"
    assert negative_zero["kind"] == "float"
    assert positive_zero["negative_zero"] is False
    assert negative_zero["negative_zero"] is True
    assert positive_zero != negative_zero


def test_lossless_semantic_values_serialize_decimal_datetime_and_binary():
    values = [
        encode_semantic_value(Decimal("1.2300"), observed_type="decimal128(10, 4)"),
        encode_semantic_value(
            datetime(2026, 7, 13, 9, 30, tzinfo=timezone.utc),
            observed_type="timestamp[us, tz=UTC]",
        ),
        encode_semantic_value(b"\x00\xff", observed_type="binary"),
    ]

    assert values[0]["kind"] == "decimal"
    assert values[0]["value"] == "1.2300"
    assert values[0]["exponent"] == -4
    assert values[1]["kind"] == "datetime"
    assert values[1]["timezone"] == "UTC"
    assert values[2] == {
        "kind": "binary",
        "hex": "00ff",
        "runtime_type": "bytes",
        "observed_type": "binary",
    }
    json.dumps(values, ensure_ascii=False, sort_keys=True)


def test_normalizer_collapses_near_integer_float_after_rounding():
    assert _norm_value(5.000000000000001) == 5
    assert type(_norm_value(5.000000000000001)) is int


def test_normalizer_preserves_lossless_rows_beside_legacy_normalized_rows():
    df = pd.DataFrame(
        {
            "missing": pd.Series([None, float("nan")], dtype=object),
            "zero": pd.Series([-0.0, 0.0], dtype=object),
            "near": pd.Series([5.000000000000001, 5.0], dtype=object),
        }
    )

    result = normalize_result(BackendResult("pandas", "ok", data=df), Program("prog", 1, []))

    assert result.lossless_schema_version == LOSSLESS_VALUE_SCHEMA_VERSION
    assert result.column_types == ["object", "object", "object"]
    assert result.logical_column_types == ["opaque:storage=object"] * 3
    assert result.columns == ["missing", "near", "zero"]
    assert all(row[0] is None for row in result.rows)
    assert all(row[1] == 5 for row in result.rows)
    missing_values = [row[0] for row in result.lossless_rows]
    near_values = [row[1] for row in result.lossless_rows]
    zero_values = [row[2] for row in result.lossless_rows]
    assert {value["kind"] for value in missing_values} == {"null", "nan"}
    assert len({value["hex"] for value in near_values}) == 2
    assert {value["negative_zero"] for value in zero_values} == {False, True}

    restored = NormalizedResult.from_dict(result.to_dict())
    assert restored.column_types == result.column_types
    assert restored.logical_column_types == result.logical_column_types
    assert restored.lossless_rows == result.lossless_rows
    assert restored.lossless_ordered_signature == result.lossless_ordered_signature


def test_normalized_result_loads_legacy_payload_with_derived_lossless_view():
    result = NormalizedResult.from_dict(
        {
            "backend": "legacy",
            "status": "ok",
            "columns": ["x"],
            "rows": [[1]],
        }
    )

    assert result.lossless_rows == []
    assert result.lossless_schema_version == ""
    assert result.effective_lossless_rows[0][0]["kind"] == "int"
    assert result.effective_lossless_rows[0][0]["value"] == "1"


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
    assert result.stable_row_keys == [canonical_key(row) for row in result.rows]
    assert result.ordered_row_signature == canonical_key(result.stable_row_keys)
    assert result.unordered_row_signature == result.ordered_row_signature


def test_normalizer_comparison_key_reuses_object_level_row_signature():
    df = pd.DataFrame([[2], [1], [1]], columns=["x"])

    result = normalize_result(BackendResult("pandas", "ok", data=df), Program("prog", 1, []))

    assert result.rows == [[1], [1], [2]]
    assert result.has_duplicate_rows is True
    assert result.comparison_key == result_comparison_key(
        status="ok",
        columns=["x"],
        ordered_row_signature=result.ordered_row_signature,
        error_type="",
    )


def test_normalizer_preserves_explicit_sort_order_for_order_sensitive_programs():
    df = pd.DataFrame([[2], [1]], columns=["x"])
    program = Program("prog", 1, [{"op": "sort", "columns": ["x"], "ascending": False}])

    result = normalize_result(BackendResult("pandas", "ok", data=df), program)

    assert program.order_sensitive is True
    assert result.rows == [[2], [1]]


def test_normalizer_preserves_float_precision_for_order_sensitive_programs():
    df = pd.DataFrame([[1.6], [1.5999999999999999]], columns=["x"])
    program = Program("prog", 1, [{"op": "sort", "columns": ["x"], "ascending": False}])

    result = normalize_result(BackendResult("pandas", "ok", data=df), program)

    assert program.order_sensitive is True
    assert result.rows == [[1.6], [1.5999999999999999]]
    assert result.rows[0] != result.rows[1]


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


def test_order_sensitive_survives_row_preserving_ops_after_sort():
    program = Program(
        "prog",
        1,
        [
            {"op": "sort", "columns": ["x"], "ascending": False},
            {"op": "filter", "column": "flag", "cmp": "==", "value": True},
            {"op": "mutate", "column": "m", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "select", "columns": ["m"]},
            {"op": "limit", "n": 1},
        ],
    )

    assert program.order_sensitive is True


def test_internal_running_sum_order_is_not_final_order_after_join():
    program = Program(
        "prog-running-join",
        1,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "order_by": [{"column": "x", "ascending": True, "nulls": "last"}],
            },
            {"op": "join", "table": "t1", "left_on": "x", "right_on": "x", "how": "inner"},
        ],
    )
    df = pd.DataFrame([[2], [1]], columns=["x"])

    result = normalize_result(BackendResult("pandas", "ok", data=df), program)

    assert program.order_sensitive is True
    assert program.output_order_sensitive is False
    assert result.rows == [[1], [2]]


def test_explicit_sort_after_join_defines_final_output_order():
    program = Program(
        "prog-running-join-sort",
        1,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "order_by": [{"column": "x", "ascending": True, "nulls": "last"}],
            },
            {"op": "join", "table": "t1", "left_on": "x", "right_on": "x", "how": "inner"},
            {"op": "sort", "columns": ["x"], "ascending": True},
        ],
    )

    assert program.output_order_sensitive is True


def test_sortedness_check_marks_program_order_sensitive():
    program = Program(
        "prog",
        1,
        [
            {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
            {"op": "sortedness_check", "column": "x", "as": "sorted_ok_x", "ascending": True, "nulls": "first"},
        ],
    )

    assert program.order_sensitive is True
