import warnings

import pytest

from datadiff.backends.pandas_backend import _records_without_duplicate_column_warning


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
