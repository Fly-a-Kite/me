from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any

from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.contract_universe import operation_type_domain
from datadiff.dsl import Case
from datadiff.experiment_manifest import stable_digest
from datadiff.semantic_core.contract_coverage import ContractCellKey


CONTRACT_TEST_EVIDENCE_SCHEMA_VERSION = "contract-test-evidence-v1"


def build_p6_contract_test_evidence(
    result: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Extract exact tested cells from successful frozen P6 executions.

    A cell means that its operation token and logical type co-occurred in one
    successfully executed case under the recorded semantic, mode, and layout
    axes. It does not infer unobserved Cartesian cells from capability claims.
    """

    target_by_backend = {
        str(target.get("backend", target.get("name", ""))): target
        for target in targets
    }
    backend_rows = result.get("backend_audit", {}).get("rows", [])
    cases = {
        str(row["case_id"]): _case_axes(row)
        for row in backend_rows
        if isinstance(row, Mapping) and isinstance(row.get("case"), Mapping)
    }
    execution_records: list[dict[str, Any]] = []
    cell_executions: dict[ContractCellKey, set[str]] = defaultdict(set)

    def record_execution(
        *,
        section: str,
        backend: str,
        case_id: str,
        status: str,
        mode: str,
        layout: str,
    ) -> None:
        if status != "ok" or backend not in target_by_backend or case_id not in cases:
            return
        target = target_by_backend[backend]
        model = target.get("capability_model", {})
        if not isinstance(model, Mapping):
            return
        case_axes = cases[case_id]
        type_support = _type_support(model)
        operation_type_pairs = tuple(
            pair
            for pair in case_axes["operation_type_pairs"]
            if pair[0] in _axis(model, "operation_tokens")
            and pair[1] in type_support.get(pair[0], ())
        )
        null_policies = _observed_null_policies(model, case_axes)
        order_policies = _observed_order_policies(model, case_axes)
        if mode not in _axis(model, "execution_modes"):
            return
        if layout not in _axis(model, "physical_layouts"):
            return
        if not all((operation_type_pairs, null_policies, order_policies)):
            return
        execution_id = stable_digest(
            "contract-execution",
            {
                "section": section,
                "backend": backend,
                "case_id": case_id,
                "mode": mode,
                "layout": layout,
            },
            length=20,
        )
        execution_records.append(
            {
                "execution_id": execution_id,
                "section": section,
                "backend": backend,
                "case_id": case_id,
                "status": status,
                "operation_type_pairs": [list(pair) for pair in operation_type_pairs],
                "null_policies": list(null_policies),
                "order_policies": list(order_policies),
                "execution_mode": mode,
                "physical_layout": layout,
            }
        )
        for (operation, logical_type), null_policy, order_policy in product(
            operation_type_pairs,
            null_policies,
            order_policies,
        ):
            key: ContractCellKey = (
                backend,
                operation,
                logical_type,
                null_policy,
                order_policy,
                mode,
                layout,
            )
            cell_executions[key].add(execution_id)

    for row in backend_rows:
        if not isinstance(row, Mapping):
            continue
        case_id = str(row.get("case_id", ""))
        raw = row.get("raw", {})
        if not isinstance(raw, Mapping):
            continue
        for backend, outcome in raw.items():
            target = target_by_backend.get(str(backend), {})
            model = target.get("capability_model", {}) if isinstance(target, Mapping) else {}
            modes = _axis(model, "execution_modes") if isinstance(model, Mapping) else ()
            layouts = _axis(model, "physical_layouts") if isinstance(model, Mapping) else ()
            if len(modes) != 1 or len(layouts) != 1:
                continue
            record_execution(
                section="backend_audit",
                backend=str(backend),
                case_id=case_id,
                status=_outcome_status(outcome),
                mode=modes[0],
                layout=layouts[0],
            )

    mode_audits = result.get("mode_audits", {})
    if isinstance(mode_audits, Mapping):
        for audit_name, audit in mode_audits.items():
            if not isinstance(audit, Mapping):
                continue
            for row in audit.get("rows", []):
                if not isinstance(row, Mapping):
                    continue
                case_id = str(row.get("case_id", ""))
                raw = row.get("raw", {})
                if not isinstance(raw, Mapping):
                    continue
                for backend, outcome in raw.items():
                    target = target_by_backend.get(str(backend), {})
                    model = (
                        target.get("capability_model", {})
                        if isinstance(target, Mapping)
                        else {}
                    )
                    modes = _axis(model, "execution_modes") if isinstance(model, Mapping) else ()
                    layouts = _axis(model, "physical_layouts") if isinstance(model, Mapping) else ()
                    if len(modes) != 1 or len(layouts) != 1:
                        continue
                    record_execution(
                        section=f"mode_audits.{audit_name}",
                        backend=str(backend),
                        case_id=case_id,
                        status=_outcome_status(outcome),
                        mode=modes[0],
                        layout=layouts[0],
                    )

    layout_audit = result.get("layout_audit", {})
    if isinstance(layout_audit, Mapping):
        for row in layout_audit.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            case_id = str(row.get("case_id", ""))
            raw = row.get("raw", {})
            if not isinstance(raw, Mapping):
                continue
            for outcome_name, outcome in raw.items():
                prefix = "pyarrow_"
                if not str(outcome_name).startswith(prefix):
                    continue
                record_execution(
                    section="layout_audit",
                    backend="pyarrow",
                    case_id=case_id,
                    status=_outcome_status(outcome),
                    mode="arrow_compute",
                    layout=str(outcome_name)[len(prefix) :],
                )

    execution_records.sort(key=lambda row: row["execution_id"])
    cells = [
        {
            "cell_key": list(key),
            "evidence_count": len(execution_ids),
            "execution_ids": sorted(execution_ids),
        }
        for key, execution_ids in sorted(cell_executions.items())
    ]
    backend_counts = Counter(row["backend"] for row in execution_records)
    payload = {
        "schema_version": CONTRACT_TEST_EVIDENCE_SCHEMA_VERSION,
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "result_digest": str(result.get("result_digest", "")),
        },
        "construction_rule": (
            "successful execution with operation/type pairs derived from CCS-IR direct "
            "column reads, scalar inputs, aggregate inputs, or row-context operations"
        ),
        "limitations": [
            "limit/offset and other row-context operations bind to current output schema types",
            "adapter conformance rows without a serialized case are excluded",
            "failed, missing, skipped, or unsupported executions do not mark cells tested",
        ],
        "summary": {
            "source_case_count": len(cases),
            "successful_execution_count": len(execution_records),
            "tested_cell_count": len(cells),
            "backend_execution_counts": dict(sorted(backend_counts.items())),
        },
        "target_capability_digests": {
            backend: str(target.get("capability_model", {}).get("digest", ""))
            for backend, target in sorted(target_by_backend.items())
        },
        "executions": execution_records,
        "cells": cells,
    }
    payload["evidence_digest"] = stable_digest("contract-test-evidence", payload)
    return payload


def _case_axes(row: Mapping[str, Any]) -> dict[str, Any]:
    required = tuple(sorted({str(token) for token in row.get("required_capabilities", [])}))
    case_payload = row["case"]
    case = Case.from_dict(dict(case_payload))
    ir = case_to_ccs_ir(case)
    values = [
        value
        for table in case_payload.get("tables", [])
        for data_row in table.get("rows", [])
        for value in data_row.values()
    ]
    return {
        "operation_tokens": tuple(
            token for token in required if not token.startswith("type:") and token != "nulls"
        ),
        "logical_types": tuple(
            token.split(":", 1)[1] for token in required if token.startswith("type:")
        ),
        "operation_type_pairs": _operation_type_pairs(ir, required),
        "has_null": any(value is None for value in values),
        "has_nan": any(isinstance(value, float) and math.isnan(value) for value in values),
        "has_explicit_sort": "op:sort" in required,
    }


def _operation_type_pairs(ir: Any, required: Sequence[str]) -> tuple[tuple[str, str], ...]:
    required_operations = {
        token for token in required if not token.startswith("type:") and token != "nulls"
    }
    pairs: set[tuple[str, str]] = set()
    source_types = {
        column.logical_type
        for relation in ir.source_relations
        for column in relation.columns
    }
    for operation in required_operations:
        if operation.startswith("table:"):
            _add_supported_pairs(pairs, operation, source_types)
    for node in ir.nodes:
        node_operations = {
            token
            for token in node.required_capabilities
            if token in required_operations
        }
        read_types = {column.logical_type for column in node.column_reads}
        output_types = {column.logical_type for column in node.output_relation.columns}
        for operation in node_operations:
            candidate_types = set(read_types)
            if operation.startswith("expr:"):
                expression_kind = operation.split(":", 1)[1]
                candidate_types = {
                    input_column.logical_type
                    for expression in node.scalar_expressions
                    if expression.kind == expression_kind
                    for input_column in expression.inputs
                }
            elif operation.startswith("agg:"):
                function = operation.split(":", 1)[1]
                candidate_types = {
                    aggregate.input_column.logical_type
                    for aggregate in node.aggregates
                    if aggregate.function == function
                    and aggregate.input_column is not None
                }
            if not candidate_types:
                candidate_types = set(output_types)
            _add_supported_pairs(pairs, operation, candidate_types)
    return tuple(sorted(pairs))


def _add_supported_pairs(
    pairs: set[tuple[str, str]],
    operation: str,
    logical_types: Iterable[str],
) -> None:
    domain = set(operation_type_domain(operation))
    pairs.update(
        (operation, str(logical_type))
        for logical_type in logical_types
        if str(logical_type) in domain
    )


def _observed_null_policies(
    model: Mapping[str, Any],
    case_axes: Mapping[str, Any],
) -> tuple[str, ...]:
    observed: list[str] = []
    for policy in _axis(model, "null_semantics"):
        if policy == "nan_distinct_from_null":
            if case_axes["has_nan"]:
                observed.append(policy)
        elif case_axes["has_null"]:
            observed.append(policy)
    return tuple(observed)


def _observed_order_policies(
    model: Mapping[str, Any],
    case_axes: Mapping[str, Any],
) -> tuple[str, ...]:
    declared = _axis(model, "order_semantics")
    if case_axes["has_explicit_sort"] and "explicit_sort" in declared:
        return ("explicit_sort",)
    if len(declared) == 1:
        return declared
    return tuple(policy for policy in declared if policy != "explicit_sort")


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
        str(operation): tuple(str(value) for value in logical_types)
        for operation, logical_types in payload.items()
        if isinstance(logical_types, (list, tuple, set, frozenset))
    }


def _outcome_status(outcome: Any) -> str:
    return str(outcome.get("status", "")) if isinstance(outcome, Mapping) else ""
