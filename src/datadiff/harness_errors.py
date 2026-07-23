"""Stable recognition of errors raised by DataDiff's own execution harness."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def harness_lowering_errors(raw_results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return backends whose observation records a shared lowering failure."""

    failures: list[str] = []
    for backend, raw in raw_results.items():
        error_type = str(raw.get("error_type", "") or "")
        error = str(raw.get("error", "") or "")
        if error_type == "HarnessLoweringError" or "HarnessLoweringError" in error:
            failures.append(str(backend))
    return sorted(failures)


def lowering_failure_classification(
    backends: list[str],
    *,
    classify: Callable[..., Any],
    build_adjudication: Callable[..., dict[str, Any]],
) -> Any:
    """Build the one fail-closed classification shared by all SQL harnesses."""

    return classify(
        "harness_lowering_error",
        "exclude_harness_lowering_failure",
        "high",
        false_positive=True,
        false_positive_reason="shared_sql_lowering_failure",
        evidence=(
            "A DataDiff SQL-lowering invariant failed before backend semantics "
            f"were comparable: {', '.join(backends)}"
        ),
        recommendation=[
            "Do not classify this as a semantic boundary or backend divergence.",
            "Minimize the lowering sequence and repair the shared SQL harness first.",
        ],
        adjudication=build_adjudication(
            "harness_lowering_error",
            validity_gate="harness_failure",
            semantic_gate="out_of_scope",
            attribution_gate="shared_sql_lowering_fault",
            exclusion_reason="shared_sql_lowering_failure",
        ),
    )
