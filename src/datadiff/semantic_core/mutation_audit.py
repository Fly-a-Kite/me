from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from datadiff.canonicalization import canonical_key, compare_result_batch
from datadiff.contract_comparison import (
    comparison_payload_for_case,
    compare_results_under_contract,
)
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.experiment_manifest import stable_digest
from datadiff.normalizer import NormalizedResult
from datadiff.semantic_values import (
    LOSSLESS_VALUE_SCHEMA_VERSION,
    encode_semantic_value,
)


SEMANTIC_MUTATION_AUDIT_SCHEMA_VERSION = "semantic-mutation-audit-v1"
PayloadMutator = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class _MutationWitness:
    mutant_id: str
    risk: str
    case: Case
    results: tuple[NormalizedResult, ...]
    mutate_payloads: PayloadMutator


def audit_semantic_mutations() -> dict[str, Any]:
    """Run deterministic high-risk weakening operators against fixed witnesses.

    This is a semantic operator mutation audit, not source-level line mutation.
    A mutant is killed only when the production comparator rejects its witness
    while the deliberately weakened comparison accepts the same observations.
    """

    rows: list[dict[str, Any]] = []
    for witness in _mutation_witnesses():
        production = compare_results_under_contract(
            witness.case,
            witness.results,
        )
        payloads = [
            comparison_payload_for_case(
                witness.case,
                result,
                profile=production.profile,
            )
            for result in witness.results
        ]
        weakened_payloads = witness.mutate_payloads(deepcopy(payloads))
        weakened = compare_result_batch(weakened_payloads)
        killed = production.has_mismatch and not weakened.has_mismatch
        row = {
            "mutant_id": witness.mutant_id,
            "risk": witness.risk,
            "case_id": witness.case.case_id,
            "production_mismatch": production.has_mismatch,
            "production_mismatch_class": production.mismatch_class,
            "weakened_comparator_accepted": not weakened.has_mismatch,
            "weakened_mismatch_class": weakened.mismatch_class,
            "killed": killed,
            "witness_digest": stable_digest(
                "semantic-mutation-witness",
                {
                    "case": witness.case.to_dict(),
                    "results": [result.to_dict() for result in witness.results],
                    "mutant_id": witness.mutant_id,
                },
            ),
        }
        rows.append(row)

    killed_count = sum(int(row["killed"]) for row in rows)
    payload = {
        "schema_version": SEMANTIC_MUTATION_AUDIT_SCHEMA_VERSION,
        "audit_kind": "deterministic_semantic_operator_mutation",
        "source_level_mutation": False,
        "summary": {
            "high_risk_mutant_count": len(rows),
            "killed_count": killed_count,
            "survived_count": len(rows) - killed_count,
            "kill_rate": killed_count / len(rows) if rows else None,
            "all_high_risk_mutants_killed": bool(rows) and killed_count == len(rows),
        },
        "mutants": rows,
    }
    payload["audit_digest"] = stable_digest("semantic-mutation-audit", payload)
    return payload


def _mutation_witnesses() -> tuple[_MutationWitness, ...]:
    return (
        _MutationWitness(
            mutant_id="collapse_nan_into_null",
            risk="NaN/NULL distinction is erased",
            case=_comparison_case(
                "nan-null",
                input_value=float("nan"),
            ),
            results=(
                _lossless_result("null", None, encode_semantic_value(None)),
                _lossless_result("nan", None, encode_semantic_value(float("nan"))),
            ),
            mutate_payloads=_collapse_nan_into_null,
        ),
        _MutationWitness(
            mutant_id="collapse_negative_zero",
            risk="IEEE-754 signed zero distinction is erased",
            case=_comparison_case("signed-zero"),
            results=(
                _lossless_result("positive", 0.0, encode_semantic_value(0.0)),
                _lossless_result("negative", -0.0, encode_semantic_value(-0.0)),
            ),
            mutate_payloads=_collapse_negative_zero,
        ),
        _MutationWitness(
            mutant_id="ignore_ordered_row_order",
            risk="an ordered result is compared as a bag",
            case=_comparison_case(
                "row-order",
                operations=[{"op": "sort", "columns": ["x"], "ascending": True}],
            ),
            results=(
                _rows_result("ordered", [[1], [2]]),
                _rows_result("reversed", [[2], [1]]),
            ),
            mutate_payloads=_sort_all_rows,
        ),
        _MutationWitness(
            mutant_id="drop_duplicate_multiplicity",
            risk="bag multiplicity is reduced to set membership",
            case=_comparison_case("multiplicity"),
            results=(
                _rows_result("duplicate", [[1], [1]]),
                _rows_result("single", [[1]]),
            ),
            mutate_payloads=_deduplicate_all_rows,
        ),
        _MutationWitness(
            mutant_id="ignore_exact_dtype",
            risk="an exact contract ignores logical dtype differences",
            case=_comparison_case(
                "dtype",
                metadata={"semantic_comparison": {"view": "exact"}},
            ),
            results=(
                _rows_result("integer", [[1]], column_type="Int64"),
                _rows_result("floating", [[1]], column_type="float64"),
            ),
            mutate_payloads=_remove_dtype_evidence,
        ),
    )


def _comparison_case(
    suffix: str,
    *,
    input_value: Any = 1.0,
    operations: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Case:
    return Case(
        f"mutation-{suffix}",
        1,
        [TableData("t0", [ColumnSpec("x", "float")], [{"x": input_value}])],
        Program(
            f"mutation-program-{suffix}",
            1,
            operations or [{"op": "select", "columns": ["x"]}],
        ),
        metadata=metadata or {},
    )


def _lossless_result(
    backend: str,
    value: Any,
    semantic_value: Mapping[str, Any],
) -> NormalizedResult:
    return NormalizedResult(
        backend,
        "ok",
        ["x"],
        [[value]],
        column_types=["float64"],
        lossless_rows=[[dict(semantic_value)]],
        lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
    )


def _rows_result(
    backend: str,
    rows: Sequence[Sequence[Any]],
    *,
    column_type: str = "int64",
) -> NormalizedResult:
    return NormalizedResult(
        backend,
        "ok",
        ["x"],
        [list(row) for row in rows],
        column_types=[column_type],
    )


def _collapse_nan_into_null(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_rewrite_values(payload, _nan_to_null) for payload in payloads]


def _collapse_negative_zero(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_rewrite_values(payload, _negative_zero_to_zero) for payload in payloads]


def _sort_all_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for payload in payloads:
        payload["rows"] = sorted(list(payload.get("rows", [])), key=canonical_key)
    return payloads


def _deduplicate_all_rows(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for payload in payloads:
        unique: dict[str, Any] = {}
        for row in list(payload.get("rows", [])):
            unique.setdefault(canonical_key(row), row)
        payload["rows"] = [unique[key] for key in sorted(unique)]
    return payloads


def _remove_dtype_evidence(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for payload in payloads:
        payload["columns"] = [
            str(column).split("\u0000", 1)[0]
            for column in list(payload.get("columns", []))
        ]
    return [_rewrite_values(payload, _drop_logical_type) for payload in payloads]


def _rewrite_values(value: Any, rewrite: Callable[[Any], Any]) -> Any:
    rewritten = rewrite(value)
    if rewritten is not value:
        return rewritten
    if isinstance(value, dict):
        return {
            key: _rewrite_values(item, rewrite)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_values(item, rewrite) for item in value]
    return value


def _nan_to_null(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "nan":
        return {"kind": "null"}
    return value


def _negative_zero_to_zero(value: Any) -> Any:
    if (
        isinstance(value, dict)
        and value.get("kind") == "number"
        and value.get("value") == "-0"
    ):
        return {**value, "value": "0"}
    return value


def _drop_logical_type(value: Any) -> Any:
    if isinstance(value, dict) and "logical_type" in value:
        return {key: item for key, item in value.items() if key != "logical_type"}
    return value
