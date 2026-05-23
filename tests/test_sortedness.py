import pandas as pd

from datadiff.sortedness import is_sorted_values


def test_sortedness_treats_pandas_na_as_null_like():
    assert is_sorted_values([1, 2, pd.NA], ascending=True, nulls="last") is True
    assert is_sorted_values([1, 2, pd.NA], ascending=True, nulls="first") is False
