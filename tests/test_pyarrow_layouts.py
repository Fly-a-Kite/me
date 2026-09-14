"""Regression tests for the extended PyArrow physical layouts.

The Boolean hash-aggregate corpus case is the reproducer for the sliced-layout bug
fixed in PyArrow 25.0.0, so every supported layout must now agree with the
contiguous layout on that case.
"""

from __future__ import annotations

import json
from pathlib import Path

from datadiff.backends import make_backend
from datadiff.dsl import Case
from datadiff.normalizer import NormalizedResult, normalize_result

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = (
    REPO_ROOT
    / "experiments/canonical_confirmed_bug_corpus/v1/cases/pyarrow_sliced_bool_groupby.json"
)
LAYOUTS = ("sliced", "chunked", "dictionary", "large_string", "run_end")


def _run(layout: str) -> NormalizedResult:
    case = Case.from_dict(json.loads(CASE_PATH.read_text(encoding="utf-8")))
    backend = make_backend("pyarrow")
    backend.configure_physical_layout(layout)
    result = backend.run(list(case.tables), case.program, timeout_s=30.0)
    return normalize_result(result, case.program)


def test_pyarrow_layouts_declared() -> None:
    backend = make_backend("pyarrow")
    for layout in ("contiguous", *LAYOUTS):
        assert layout in backend.supported_physical_layouts


def test_pyarrow_extended_layouts_match_contiguous() -> None:
    baseline = _run("contiguous")
    assert baseline.status == "ok", baseline.error
    for layout in LAYOUTS:
        normalized = _run(layout)
        assert normalized.status == "ok", f"{layout}: {normalized.error}"
        assert normalized.comparison_key == baseline.comparison_key, layout
