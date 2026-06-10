import warnings

import pytest

from datadiff.backends.pandas_backend import _records_without_duplicate_column_warning
from datadiff.backends.pandas_backend import PandasBackend
from datadiff.datagen import generate_case


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
