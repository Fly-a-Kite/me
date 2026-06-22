from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from datadiff.adjudication import build_adjudication
from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.dsl import Case
from datadiff.filtering import evaluate_filter_predicate
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding

WITNESS_CONTRACT_SCHEMA = "datadiff-witness-contract-v1"
WITNESS_ORACLE_SCHEMA = "datadiff-witness-oracle-result-v1"
REFERENCE_WITNESS_PLAN_SCHEMA = "datadiff-reference-witness-plan-v1"
SUPPORTED_WITNESS_KINDS = frozenset(
    {
        "row_containment",
        "row_absence",
        "group_containment",
        "aggregate_value",
    }
)


@dataclass(frozen=True, slots=True)
class WitnessContract:
    kind: str
    row: dict[str, Any] = field(default_factory=dict)
    group_key: dict[str, Any] = field(default_factory=dict)
    aggregate: str = ""
    expected: Any = None
    predicate: dict[str, Any] | None = None
    backends: list[str] = field(default_factory=list)
    reason: str = ""
    root_cause: str = ""
    family: str = ""
    source: str = "explicit"
    schema_version: str = WITNESS_CONTRACT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.predicate is None:
            payload.pop("predicate", None)
        if self.expected is None:
            payload.pop("expected", None)
        if not self.root_cause:
            payload.pop("root_cause", None)
        if not self.family:
            payload.pop("family", None)
        return payload


@dataclass(frozen=True, slots=True)
class WitnessBackendCheck:
    backend: str
    status: str
    satisfied: bool
    matched_count: int = 0
    expected: Any = None
    actual: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WitnessOracleResult:
    schema_version: str
    enabled: bool
    contract_present: bool
    contract: dict[str, Any]
    checks: list[WitnessBackendCheck]
    failing_backends: list[str]
    satisfied_backends: list[str]
    unsupported_reason: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.contract_present) and not self.unsupported_reason and not self.failing_backends

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "contract_present": self.contract_present,
            "contract": self.contract,
            "checks": [check.to_dict() for check in self.checks],
            "failing_backends": list(self.failing_backends),
            "satisfied_backends": list(self.satisfied_backends),
            "unsupported_reason": self.unsupported_reason,
            "passed": self.passed,
        }


def witness_contract_from_case(case: Case, *, infer: bool = False) -> WitnessContract | None:
    payload = case.metadata.get("witness_contract") if isinstance(case.metadata, dict) else None
    if isinstance(payload, Mapping):
        return coerce_witness_contract(payload)
    return infer_witness_contract(case) if infer else None


def coerce_witness_contract(payload: Mapping[str, Any]) -> WitnessContract:
    kind = str(payload.get("kind", "") or "").strip()
    row = _string_key_mapping(payload.get("row", {}))
    group_key = _string_key_mapping(payload.get("group_key", payload.get("key", {})))
    backends = [str(item) for item in payload.get("backends", []) or [] if str(item)]
    predicate = payload.get("predicate")
    return WitnessContract(
        schema_version=str(payload.get("schema_version", WITNESS_CONTRACT_SCHEMA) or WITNESS_CONTRACT_SCHEMA),
        kind=kind,
        row=row,
        group_key=group_key,
        aggregate=str(payload.get("aggregate", payload.get("column", "")) or ""),
        expected=payload.get("expected"),
        predicate=dict(predicate) if isinstance(predicate, Mapping) else None,
        backends=backends,
        reason=str(payload.get("reason", "") or ""),
        root_cause=str(payload.get("root_cause", "") or ""),
        family=str(payload.get("family", "") or ""),
        source=str(payload.get("source", "explicit") or "explicit"),
    )


def build_reference_witness_plan(
    normalized: Mapping[str, Mapping[str, Any] | NormalizedResult],
    *,
    suspicious_backends: list[str],
    reference_backends: list[str],
    operations: list[Mapping[str, Any]] | None = None,
    root_cause: str = "",
    family: str = "",
) -> dict[str, Any]:
    """Infer a PQS-style witness fact from reference-backend consensus.

    This is intentionally a triage/reproducer helper. It creates a witness only
    when the listed reference backends all satisfy the fact and at least one
    suspicious backend violates it.
    """
    coerced = _coerce_normalized_mapping(normalized)
    suspicious = [backend for backend in unique_backend_names(suspicious_backends) if backend in coerced]
    references = [
        backend
        for backend in unique_backend_names(reference_backends)
        if backend in coerced and coerced[backend].status == "ok"
    ]
    base = {
        "schema_version": REFERENCE_WITNESS_PLAN_SCHEMA,
        "source": "reference_consensus_from_existing_queue_outputs",
        "policy": "triage_and_reproducer_only_not_authority_count",
        "family": str(family),
        "root_cause": str(root_cause),
        "suspicious_backends": suspicious,
        "reference_backends": references,
    }
    if not suspicious:
        return {**base, "status": "not_available", "reason": "missing_suspicious_backend_output"}
    if not references:
        return {**base, "status": "not_available", "reason": "missing_ok_reference_backend_output"}
    contract = infer_reference_witness_contract(
        coerced,
        suspicious_backends=suspicious,
        reference_backends=references,
        operations=operations or [],
        root_cause=root_cause,
        family=family,
    )
    if contract is None:
        return {**base, "status": "not_available", "reason": "no_reference_consensus_fact_violated_by_suspicious_backend"}
    checks = [_check_backend(contract, backend, coerced.get(backend)) for backend in contract.backends]
    failing = sorted(check.backend for check in checks if not check.satisfied)
    satisfied = sorted(check.backend for check in checks if check.satisfied)
    return {
        **base,
        "status": "available",
        "contract": contract.to_dict(),
        "satisfied_reference_backends": sorted(backend for backend in references if backend in satisfied),
        "failing_suspicious_backends": sorted(backend for backend in suspicious if backend in failing),
        "checks": [check.to_dict() for check in checks],
    }


def infer_reference_witness_contract(
    normalized: Mapping[str, Mapping[str, Any] | NormalizedResult],
    *,
    suspicious_backends: list[str],
    reference_backends: list[str],
    operations: list[Mapping[str, Any]] | None = None,
    root_cause: str = "",
    family: str = "",
) -> WitnessContract | None:
    coerced = _coerce_normalized_mapping(normalized)
    suspicious = [backend for backend in unique_backend_names(suspicious_backends) if backend in coerced]
    references = [
        backend
        for backend in unique_backend_names(reference_backends)
        if backend in coerced and coerced[backend].status == "ok"
    ]
    if not suspicious or not references:
        return None
    group_keys = _last_group_keys(operations or [], columns=_shared_reference_columns(coerced, references))
    aggregate_columns = _aggregate_output_columns(operations or [], columns=_shared_reference_columns(coerced, references), group_keys=group_keys)
    for row in _reference_consensus_rows(coerced, references):
        for aggregate in aggregate_columns:
            contract = WitnessContract(
                kind="aggregate_value",
                group_key={key: row[key] for key in group_keys if key in row},
                aggregate=aggregate,
                expected=row.get(aggregate),
                backends=sorted({*suspicious, *references}),
                reason="reference backends agree on aggregate witness value and suspicious backend violates it",
                root_cause=str(root_cause),
                family=str(family),
                source="reference_consensus",
            )
            if _contract_separates_reference_and_suspicious(contract, coerced, references=references, suspicious=suspicious):
                return contract
    for row in _reference_consensus_rows(coerced, references):
        contract = WitnessContract(
            kind="row_containment",
            row=row,
            backends=sorted({*suspicious, *references}),
            reason="reference backends agree this output row should be present and suspicious backend omits it",
            root_cause=str(root_cause),
            family=str(family),
            source="reference_consensus",
        )
        if _contract_separates_reference_and_suspicious(contract, coerced, references=references, suspicious=suspicious):
            return contract
    for row in _suspicious_consensus_rows(coerced, suspicious):
        contract = WitnessContract(
            kind="row_absence",
            row=row,
            backends=sorted({*suspicious, *references}),
            reason="suspicious backend emits a row that all reference backends omit",
            root_cause=str(root_cause),
            family=str(family),
            source="suspicious_extra_row",
        )
        if _contract_separates_reference_and_suspicious(contract, coerced, references=references, suspicious=suspicious):
            return contract
    return None


def infer_witness_contract(case: Case) -> WitnessContract | None:
    if len(case.tables) != 1 or not case.tables[0].rows:
        return None
    rows = [dict(row) for row in case.tables[0].rows]
    for operation in case.program.operations:
        op = operation.to_dict() if hasattr(operation, "to_dict") else dict(operation)
        kind = str(op.get("op", "") or "")
        if kind == "filter":
            column = str(op.get("column", "") or "")
            if not column:
                return None
            try:
                rows = [
                    row
                    for row in rows
                    if column in row and evaluate_filter_predicate(row.get(column), op.get("cmp"), op.get("value"))
                ]
            except Exception:
                return None
            if not rows:
                return None
            continue
        if kind == "select":
            columns = [str(column) for column in op.get("columns", []) or []]
            if not columns or any(column not in rows[0] for column in columns):
                return None
            rows = [{column: row.get(column) for column in columns} for row in rows]
            continue
        if kind == "sort":
            continue
        return None
    if not rows:
        return None
    return WitnessContract(
        kind="row_containment",
        row=dict(rows[0]),
        reason="auto-inferred from single-table row-preserving filter/select/sort pipeline",
        source="inferred_row_preserving_pipeline",
    )


def evaluate_witness_contract(
    case: Case,
    normalized: Mapping[str, Mapping[str, Any] | NormalizedResult],
    *,
    enabled: bool,
    infer: bool = False,
) -> WitnessOracleResult:
    contract = witness_contract_from_case(case, infer=infer)
    if contract is None:
        return WitnessOracleResult(
            schema_version=WITNESS_ORACLE_SCHEMA,
            enabled=bool(enabled),
            contract_present=False,
            contract={},
            checks=[],
            failing_backends=[],
            satisfied_backends=[],
        )
    if contract.kind not in SUPPORTED_WITNESS_KINDS:
        return WitnessOracleResult(
            schema_version=WITNESS_ORACLE_SCHEMA,
            enabled=bool(enabled),
            contract_present=True,
            contract=contract.to_dict(),
            checks=[],
            failing_backends=[],
            satisfied_backends=[],
            unsupported_reason=f"unsupported_witness_kind:{contract.kind}",
        )
    normalized_results = _coerce_normalized_mapping(normalized)
    target_backends = contract.backends or sorted(str(backend) for backend in normalized_results)
    checks = [
        _check_backend(contract, str(backend), normalized_results.get(str(backend)))
        for backend in target_backends
    ]
    failing = sorted(check.backend for check in checks if not check.satisfied)
    satisfied = sorted(check.backend for check in checks if check.satisfied)
    return WitnessOracleResult(
        schema_version=WITNESS_ORACLE_SCHEMA,
        enabled=bool(enabled),
        contract_present=True,
        contract=contract.to_dict(),
        checks=checks,
        failing_backends=failing,
        satisfied_backends=satisfied,
    )


def finding_from_witness_result(
    case: Case,
    result: WitnessOracleResult,
) -> Finding | None:
    if not result.contract_present or result.unsupported_reason or not result.failing_backends:
        return None
    contract = result.contract
    kind = str(contract.get("kind", "witness_contract") or "witness_contract")
    signature = short_canonical_hash(
        {
            "case_id": case.case_id,
            "witness_contract": contract,
            "failing_backends": result.failing_backends,
            "checks": [check.to_dict() for check in result.checks],
        },
        16,
    )
    return Finding(
        finding_id=f"witness-{signature}",
        kind=f"witness_{kind}_violation",
        severity="high",
        suspicious_backends=list(result.failing_backends),
        evidence=_witness_evidence(result),
        signature=signature,
        root_cause=str(contract.get("root_cause", kind) or kind),
        oracle="witness",
        confidence="high",
        triage_verdict="candidate_implementation_bug",
        paper_status="candidate_bug_needs_external_confirmation",
        triage_confidence="medium",
        triage_evidence="Witness-level local contract failed on one or more backends; requires upstream confirmation before it counts as a confirmed bug.",
        recommendation=[
            "Reduce the case around the witness contract.",
            "Confirm whether the contract matches the target backend semantics.",
            "Use upstream confirmation/fix before counting this as a confirmed latest-version bug.",
        ],
        adjudication=build_adjudication(
            "candidate_implementation_bug",
            validity_gate="valid_case",
            semantic_gate="witness_contract",
            attribution_gate="backend_candidate_bug",
            reference_support="witness_contract",
            countable_as_bug_evidence=True,
            countable_as_valid_finding=True,
            needs_manual_review=False,
            needs_external_confirmation=True,
        ),
        mismatch_class="witness_contract",
    )


def row_satisfies_predicate(row: Mapping[str, Any], predicate: Mapping[str, Any]) -> bool:
    column = str(predicate.get("column", "") or "")
    comparator = predicate.get("cmp", predicate.get("comparator", ""))
    value = predicate.get("value")
    if not column:
        raise ValueError("witness predicate missing column")
    return bool(evaluate_filter_predicate(row.get(column), comparator, value))


def rectify_predicate_for_row(
    row: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> dict[str, Any]:
    if row_satisfies_predicate(row, predicate):
        return {
            "predicate": dict(predicate),
            "changed": False,
            "reason": "predicate_already_true_for_witness_row",
        }
    column = str(predicate.get("column", "") or "")
    if not column:
        raise ValueError("witness predicate missing column")
    return {
        "predicate": {"column": column, "cmp": "==", "value": row.get(column)},
        "changed": True,
        "reason": "replaced_with_witness_equality_predicate",
    }


def _check_backend(
    contract: WitnessContract,
    backend: str,
    normalized: NormalizedResult | None,
) -> WitnessBackendCheck:
    if normalized is None:
        return WitnessBackendCheck(
            backend=backend,
            status="missing_backend",
            satisfied=False,
            error="backend result not present",
        )
    if normalized.status != "ok":
        return WitnessBackendCheck(
            backend=backend,
            status=normalized.status,
            satisfied=False,
            error=normalized.error_type or normalized.error,
        )
    try:
        row_dicts = _result_rows_as_dicts(normalized)
        if contract.kind == "row_containment":
            matched = _matching_row_count(row_dicts, contract.row)
            return WitnessBackendCheck(
                backend=backend,
                status="ok",
                satisfied=matched > 0,
                matched_count=matched,
                expected="present",
                actual="present" if matched > 0 else "missing",
            )
        if contract.kind == "row_absence":
            matched = _matching_row_count(row_dicts, contract.row)
            return WitnessBackendCheck(
                backend=backend,
                status="ok",
                satisfied=matched == 0,
                matched_count=matched,
                expected="absent",
                actual="present" if matched > 0 else "absent",
            )
        if contract.kind == "group_containment":
            matched = _matching_row_count(row_dicts, contract.group_key)
            return WitnessBackendCheck(
                backend=backend,
                status="ok",
                satisfied=matched > 0,
                matched_count=matched,
                expected="group_present",
                actual="group_present" if matched > 0 else "group_missing",
            )
        if contract.kind == "aggregate_value":
            actual = _aggregate_value(row_dicts, contract.group_key, contract.aggregate)
            expected = contract.expected
            return WitnessBackendCheck(
                backend=backend,
                status="ok",
                satisfied=_values_equal(actual, expected),
                matched_count=_matching_row_count(row_dicts, contract.group_key),
                expected=expected,
                actual=actual,
            )
    except Exception as exc:
        return WitnessBackendCheck(
            backend=backend,
            status="witness_error",
            satisfied=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return WitnessBackendCheck(
        backend=backend,
        status="unsupported",
        satisfied=False,
        error=f"unsupported witness kind: {contract.kind}",
    )


def _result_rows_as_dicts(normalized: NormalizedResult) -> list[dict[str, Any]]:
    return [
        {column: row[index] for index, column in enumerate(normalized.columns)}
        for row in normalized.rows
    ]


def _coerce_normalized_mapping(
    mapping: Mapping[str, Mapping[str, Any] | NormalizedResult],
) -> dict[str, NormalizedResult]:
    result: dict[str, NormalizedResult] = {}
    for backend, payload in mapping.items():
        key = str(backend)
        if isinstance(payload, NormalizedResult):
            result[key] = payload
            continue
        if isinstance(payload, Mapping):
            result[key] = NormalizedResult.from_dict(payload, backend=key)
    return result


def unique_backend_names(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        backend = str(value).strip()
        if not backend or backend in seen:
            continue
        seen.add(backend)
        result.append(backend)
    return result


def _shared_reference_columns(normalized: Mapping[str, NormalizedResult], references: list[str]) -> list[str]:
    if not references:
        return []
    shared = set(normalized[references[0]].columns)
    for backend in references[1:]:
        shared.intersection_update(normalized[backend].columns)
    return [column for column in normalized[references[0]].columns if column in shared]


def _reference_consensus_rows(
    normalized: Mapping[str, NormalizedResult],
    references: list[str],
) -> list[dict[str, Any]]:
    columns = _shared_reference_columns(normalized, references)
    if not columns:
        return []
    common_keys: set[str] | None = None
    rows_by_key: dict[str, dict[str, Any]] = {}
    for backend in references:
        row_map: dict[str, dict[str, Any]] = {}
        for row in _project_rows(_result_rows_as_dicts(normalized[backend]), columns):
            key = canonical_key(_ordered_subset(row))
            row_map.setdefault(key, row)
        if common_keys is None:
            common_keys = set(row_map)
            rows_by_key = dict(row_map)
        else:
            common_keys.intersection_update(row_map)
    if not common_keys:
        return []
    return [rows_by_key[key] for key in sorted(common_keys)]


def _suspicious_consensus_rows(
    normalized: Mapping[str, NormalizedResult],
    suspicious: list[str],
) -> list[dict[str, Any]]:
    ok_suspicious = [backend for backend in suspicious if normalized[backend].status == "ok"]
    if not ok_suspicious:
        return []
    columns = _shared_reference_columns(normalized, ok_suspicious)
    if not columns:
        return []
    common_keys: set[str] | None = None
    rows_by_key: dict[str, dict[str, Any]] = {}
    for backend in ok_suspicious:
        row_map: dict[str, dict[str, Any]] = {}
        for row in _project_rows(_result_rows_as_dicts(normalized[backend]), columns):
            key = canonical_key(_ordered_subset(row))
            row_map.setdefault(key, row)
        if common_keys is None:
            common_keys = set(row_map)
            rows_by_key = dict(row_map)
        else:
            common_keys.intersection_update(row_map)
    if not common_keys:
        return []
    return [rows_by_key[key] for key in sorted(common_keys)]


def _project_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: row[column] for column in columns}
        for row in rows
        if all(column in row for column in columns)
    ]


def _last_group_keys(operations: list[Mapping[str, Any]], *, columns: list[str]) -> list[str]:
    for operation in reversed(operations):
        kind = str(operation.get("op", "") or "")
        if kind != "groupby":
            continue
        keys = [str(key) for key in operation.get("keys", []) or []]
        return [key for key in keys if key in columns]
    return []


def _aggregate_output_columns(
    operations: list[Mapping[str, Any]],
    *,
    columns: list[str],
    group_keys: list[str],
) -> list[str]:
    aliases: list[str] = []
    for operation in reversed(operations):
        kind = str(operation.get("op", "") or "")
        if kind not in {"groupby", "aggregate"}:
            continue
        for aggregate in operation.get("aggs", []) or []:
            if not isinstance(aggregate, Mapping):
                continue
            alias = str(aggregate.get("as", "") or aggregate.get("column", "") or "")
            if alias:
                aliases.append(alias)
        break
    if aliases:
        return [alias for alias in aliases if alias in columns and alias not in set(group_keys)]
    return []


def _contract_separates_reference_and_suspicious(
    contract: WitnessContract,
    normalized: Mapping[str, NormalizedResult],
    *,
    references: list[str],
    suspicious: list[str],
) -> bool:
    reference_checks = [_check_backend(contract, backend, normalized.get(backend)) for backend in references]
    suspicious_checks = [_check_backend(contract, backend, normalized.get(backend)) for backend in suspicious]
    return bool(reference_checks) and all(check.satisfied for check in reference_checks) and any(
        not check.satisfied for check in suspicious_checks
    )


def _matching_row_count(rows: list[dict[str, Any]], expected: Mapping[str, Any]) -> int:
    if not expected:
        return 0
    expected_key = canonical_key(_ordered_subset(expected))
    return sum(
        1
        for row in rows
        if all(column in row for column in expected)
        and canonical_key(_ordered_subset({column: row[column] for column in expected})) == expected_key
    )


def _aggregate_value(rows: list[dict[str, Any]], group_key: Mapping[str, Any], aggregate: str) -> Any:
    if not aggregate:
        raise ValueError("aggregate_value witness missing aggregate column")
    matches = [row for row in rows if _row_matches(row, group_key)]
    if not matches:
        return None
    if any(aggregate not in row for row in matches):
        return None
    values = [row.get(aggregate) for row in matches]
    if len(values) == 1:
        return values[0]
    counts = Counter(canonical_key(value) for value in values)
    most_common = counts.most_common(1)[0][0]
    for value in values:
        if canonical_key(value) == most_common:
            return value
    return values[0]


def _row_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not expected:
        return True
    return _matching_row_count([dict(row)], expected) > 0


def _ordered_subset(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): payload[key] for key in sorted(payload, key=str)}


def _values_equal(left: Any, right: Any) -> bool:
    return canonical_key(left) == canonical_key(right)


def _string_key_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _witness_evidence(result: WitnessOracleResult) -> str:
    contract = result.contract
    reason = str(contract.get("reason", "") or "")
    prefix = f"Witness contract {contract.get('kind', 'unknown')} failed for backends {result.failing_backends}"
    return f"{prefix}; reason={reason}" if reason else prefix
