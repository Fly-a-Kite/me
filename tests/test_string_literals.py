import random

import pytest

from datadiff.dsl import ColumnSpec, TableData
from datadiff.string_literals import (
    STRING_PATTERN_FALLBACKS,
    string_pattern_literal_for_column,
    string_pattern_literal_for_type,
)


def test_string_pattern_literal_for_type_rejects_non_string_type():
    with pytest.raises(ValueError):
        string_pattern_literal_for_type(random.Random(1), "int")


def test_string_pattern_literal_for_type_uses_documented_fallback_pool():
    value = string_pattern_literal_for_type(random.Random(1), "str")

    assert value in STRING_PATTERN_FALLBACKS


def test_string_pattern_literal_for_column_samples_substring_from_table_value():
    table = TableData(
        name="t",
        columns=[ColumnSpec("s", "str")],
        rows=[{"s": "alpha beta"}],
    )

    value = string_pattern_literal_for_column(random.Random(1), table, "s", "str_contains")

    assert value
    assert value in "alpha beta"
