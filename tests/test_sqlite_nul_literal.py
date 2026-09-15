"""SQLite cannot represent NUL in TEXT; the adapter must raise the harness's typed
lowering error so the case is excluded instead of being reported as a bug."""

from __future__ import annotations

import pytest

from datadiff.backends.sql_lowering import HarnessLoweringError
from datadiff.backends.sqlite_backend import _lit


def test_nul_string_literal_raises_harness_lowering_error() -> None:
    with pytest.raises(HarnessLoweringError, match="NUL"):
        _lit("a\x00z")


def test_plain_string_literal_is_quoted() -> None:
    assert _lit("a'b") == "'a''b'"
