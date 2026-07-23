from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from datadiff.experiment_manifest import stable_digest


CONTRACT_COVERAGE_SCHEMA_VERSION = "contract-coverage-matrix-v1"
CONTRACT_STATUSES = (
    "supported_and_tested",
    "supported_but_untested",
    "explicitly_unsupported",
    "unknown_or_undeclared",
)
ContractCellKey = tuple[str, str, str, str, str, str, str]


def audit_contract_coverage(
    targets: Sequence[Mapping[str, Any]],
    *,
    tested_cells: Iterable[ContractCellKey] = (),
    requested_cells: Iterable[ContractCellKey] = (),
) -> dict[str, Any]:
    """Build operation × type × NULL/order × mode/layout coverage.

    A tested cell must use the exact seven-field key returned in each matrix row.
    Missing capability declarations are reported as unknown; an omitted value from
    an otherwise explicit declaration is reported as unsupported.
    """

    tested = {tuple(str(part) for part in cell) for cell in tested_cells}
    requested = [tuple(str(part) for part in cell) for cell in requested_cells]
    if requested:
        return _audit_requested_cells(targets, requested=requested, tested=tested)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    declared_keys: set[ContractCellKey] = set()
    for target in targets:
        backend = str(target.get("backend", target.get("name", "")) or "")
        model = target.get("capability_model", {})
        if not isinstance(model, Mapping):
            model = {}
        axes = {
            "operation": _axis(model, "operation_tokens"),
            "logical_type": _axis(model, "logical_types"),
            "null": _axis(model, "null_semantics"),
            "order": _axis(model, "order_semantics"),
            "mode": _axis(model, "execution_modes"),
            "layout": _axis(model, "physical_layouts"),
        }
        type_support = _type_support(model)
        declared = all(axes.values()) and all(
            operation in type_support for operation in axes["operation"]
        )
        if not declared:
            status = "unknown_or_undeclared"
            row = {
                "backend": backend,
                "status": status,
                "missing_declarations": [
                    axis for axis, values in axes.items() if not values
                ]
                + (
                    ["operation_type_support"]
                    if not type_support
                    or any(
                        operation not in type_support
                        for operation in axes["operation"]
                    )
                    else []
                ),
                "cell_key": [backend, "", "", "", "", "", ""],
            }
            rows.append(row)
            status_counts[status] += 1
            continue
        for operation in axes["operation"]:
            for logical_type in type_support[operation]:
                for null_policy in axes["null"]:
                    for order_policy in axes["order"]:
                        for mode in axes["mode"]:
                            for layout in axes["layout"]:
                                key: ContractCellKey = (
                                    backend,
                                    operation,
                                    logical_type,
                                    null_policy,
                                    order_policy,
                                    mode,
                                    layout,
                                )
                                status = (
                                    "supported_and_tested"
                                    if key in tested
                                    else "supported_but_untested"
                                )
                                status_counts[status] += 1
                                declared_keys.add(key)
                                rows.append(
                                    {
                                        "backend": backend,
                                        "operation": operation,
                                        "logical_type": logical_type,
                                        "null_policy": null_policy,
                                        "order_policy": order_policy,
                                        "execution_mode": mode,
                                        "physical_layout": layout,
                                        "status": status,
                                        "cell_key": list(key),
                                    }
                                )
    supported = (
        status_counts["supported_and_tested"]
        + status_counts["supported_but_untested"]
    )
    outside_declared = sorted(tested - declared_keys)
    payload = {
        "schema_version": CONTRACT_COVERAGE_SCHEMA_VERSION,
        "statuses": list(CONTRACT_STATUSES),
        "target_count": len(targets),
        "summary": {
            "cell_count": len(rows),
            **{status: status_counts[status] for status in CONTRACT_STATUSES},
            "supported_executable_coverage": (
                status_counts["supported_and_tested"] / supported
                if supported
                else None
            ),
            "unknown_semantic_fallback_count": status_counts[
                "unknown_or_undeclared"
            ],
            "tested_cell_input_count": len(tested),
            "tested_outside_declared_contract_count": len(outside_declared),
        },
        "tested_outside_declared_contract": [list(key) for key in outside_declared],
        "backends": _backend_summaries(rows),
        "cells": rows,
    }
    payload["matrix_digest"] = stable_digest("contract-coverage", payload)
    return payload


def _axis(model: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = model.get(key, ())
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(sorted({str(value) for value in values if str(value)}))


def _type_support(model: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    payload = model.get("operation_type_support", {})
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(operation): tuple(
            sorted({str(value) for value in logical_types if str(value)})
        )
        for operation, logical_types in payload.items()
        if isinstance(logical_types, (list, tuple, set, frozenset))
    }


def _audit_requested_cells(
    targets: Sequence[Mapping[str, Any]],
    *,
    requested: list[ContractCellKey],
    tested: set[ContractCellKey],
) -> dict[str, Any]:
    target_by_backend = {
        str(target.get("backend", target.get("name", "")) or ""): target
        for target in targets
    }
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for key in requested:
        backend, operation, logical_type, null_policy, order_policy, mode, layout = key
        target = target_by_backend.get(backend)
        model = target.get("capability_model", {}) if isinstance(target, Mapping) else {}
        if not isinstance(model, Mapping):
            model = {}
        axes = {
            "operation": _axis(model, "operation_tokens"),
            "logical_type": _axis(model, "logical_types"),
            "null": _axis(model, "null_semantics"),
            "order": _axis(model, "order_semantics"),
            "mode": _axis(model, "execution_modes"),
            "layout": _axis(model, "physical_layouts"),
        }
        type_support = _type_support(model)
        values = {
            "operation": operation,
            "logical_type": logical_type,
            "null": null_policy,
            "order": order_policy,
            "mode": mode,
            "layout": layout,
        }
        missing_type_declaration = operation not in type_support
        if (
            target is None
            or any(not axis_values for axis_values in axes.values())
            or not type_support
            or missing_type_declaration
        ):
            status = "unknown_or_undeclared"
            reasons = [
                axis for axis, axis_values in axes.items() if not axis_values
            ]
            if not type_support or missing_type_declaration:
                reasons.append("operation_type_support")
            if not reasons:
                reasons = ["backend"]
        else:
            reasons = [
                f"{axis}:{value}"
                for axis, value in values.items()
                if value not in axes[axis]
            ]
            if operation in axes["operation"] and logical_type not in type_support.get(
                operation, ()
            ):
                reasons.append(f"operation_type:{operation}:{logical_type}")
            status = (
                "explicitly_unsupported"
                if reasons
                else "supported_and_tested"
                if key in tested
                else "supported_but_untested"
            )
        counts[status] += 1
        rows.append(
            {
                "backend": backend,
                "operation": operation,
                "logical_type": logical_type,
                "null_policy": null_policy,
                "order_policy": order_policy,
                "execution_mode": mode,
                "physical_layout": layout,
                "status": status,
                "reasons": reasons,
                "cell_key": list(key),
            }
        )
    supported = counts["supported_and_tested"] + counts["supported_but_untested"]
    requested_set = set(requested)
    outside_requested = sorted(tested - requested_set)
    declaration_conflicts = sorted(
        key
        for key, row in zip(requested, rows)
        if key in tested and row["status"] == "explicitly_unsupported"
    )
    payload = {
        "schema_version": CONTRACT_COVERAGE_SCHEMA_VERSION,
        "statuses": list(CONTRACT_STATUSES),
        "target_count": len(targets),
        "summary": {
            "cell_count": len(rows),
            **{status: counts[status] for status in CONTRACT_STATUSES},
            "supported_executable_coverage": (
                counts["supported_and_tested"] / supported if supported else None
            ),
            "unknown_semantic_fallback_count": counts["unknown_or_undeclared"],
            "tested_cell_input_count": len(tested),
            "tested_cell_not_requested_count": len(outside_requested),
            "tested_declaration_conflict_count": len(declaration_conflicts),
        },
        "tested_cells_not_requested": [list(key) for key in outside_requested],
        "tested_declaration_conflicts": [list(key) for key in declaration_conflicts],
        "backends": _backend_summaries(rows),
        "cells": rows,
    }
    payload["matrix_digest"] = stable_digest("contract-coverage", payload)
    return payload


def _backend_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts_by_backend: dict[str, Counter[str]] = {}
    for row in rows:
        backend = str(row.get("backend", ""))
        status = str(row.get("status", ""))
        counts_by_backend.setdefault(backend, Counter())[status] += 1
    summaries: list[dict[str, Any]] = []
    for backend, counts in sorted(counts_by_backend.items()):
        supported = counts["supported_and_tested"] + counts["supported_but_untested"]
        summaries.append(
            {
                "backend": backend,
                **{status: counts[status] for status in CONTRACT_STATUSES},
                "supported_executable_coverage": (
                    counts["supported_and_tested"] / supported if supported else None
                ),
            }
        )
    return summaries
