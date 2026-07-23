"""Causal candidate signatures used for feedback and root-family accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from datadiff.experiment_manifest import stable_digest


CAUSAL_SIGNATURE_SCHEMA_VERSION = "datadiff-causal-signature-v1"


@dataclass(frozen=True, slots=True)
class CausalSignature:
    minimal_ir_prefix_digest: str
    backend: str
    backend_version: str
    dialect: str
    adapter_version: str
    lowering_version: str
    difference_fingerprint: str
    operation_sequence: tuple[str, ...]
    semantic_axes: tuple[str, ...]
    reproduction_matrix: tuple[tuple[str, str], ...]
    minimized_case_digest: str
    schema_version: str = CAUSAL_SIGNATURE_SCHEMA_VERSION
    signature_digest: str = ""

    @classmethod
    def build(
        cls,
        *,
        program_ir: Mapping[str, Any] | Sequence[Any] | None,
        backend: str,
        backend_version: str = "",
        dialect: str = "",
        adapter_version: str = "",
        lowering_version: str = "",
        normalized_difference: Mapping[str, Any] | str | None = None,
        error_family: Mapping[str, Any] | str | None = None,
        operation_sequence: Sequence[str] = (),
        semantic_axes: Sequence[str] = (),
        reproduction_matrix: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        minimized_case: Mapping[str, Any] | None = None,
        prefix_length: int | None = None,
    ) -> "CausalSignature":
        prefix = _ir_prefix(program_ir, prefix_length)
        diff = normalized_difference if normalized_difference is not None else error_family or {}
        matrix = _matrix_items(reproduction_matrix)
        payload = {
            "minimal_ir_prefix_digest": stable_digest("ir-prefix", prefix),
            "backend": str(backend),
            "backend_version": str(backend_version),
            "dialect": str(dialect),
            "adapter_version": str(adapter_version),
            "lowering_version": str(lowering_version),
            "difference_fingerprint": stable_digest("difference", diff),
            "operation_sequence": [str(value) for value in operation_sequence],
            "semantic_axes": sorted({str(value) for value in semantic_axes if str(value)}),
            "reproduction_matrix": [list(value) for value in matrix],
            "minimized_case_digest": stable_digest("minimized-case", minimized_case or {}),
            "schema_version": CAUSAL_SIGNATURE_SCHEMA_VERSION,
        }
        return cls(
            minimal_ir_prefix_digest=payload["minimal_ir_prefix_digest"],
            backend=payload["backend"],
            backend_version=payload["backend_version"],
            dialect=payload["dialect"],
            adapter_version=payload["adapter_version"],
            lowering_version=payload["lowering_version"],
            difference_fingerprint=payload["difference_fingerprint"],
            operation_sequence=tuple(payload["operation_sequence"]),
            semantic_axes=tuple(payload["semantic_axes"]),
            reproduction_matrix=tuple(tuple(value) for value in matrix),
            minimized_case_digest=payload["minimized_case_digest"],
            signature_digest=stable_digest("causal-signature", payload),
        )

    @property
    def group_key(self) -> str:
        return self.signature_digest

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["operation_sequence"] = list(self.operation_sequence)
        payload["semantic_axes"] = list(self.semantic_axes)
        payload["reproduction_matrix"] = [list(value) for value in self.reproduction_matrix]
        return payload


def causal_signature_from_row(
    row: Mapping[str, Any],
    finding: Mapping[str, Any],
    *,
    backend: str,
    minimized_case: Mapping[str, Any] | None = None,
) -> CausalSignature:
    case = row.get("case", {})
    program = case.get("program", {}) if isinstance(case, Mapping) else {}
    operations = program.get("operations", ()) if isinstance(program, Mapping) else ()
    operation_sequence = [
        str(operation.get("op", operation.get("kind", "")) or "")
        for operation in operations
        if isinstance(operation, Mapping)
    ]
    target = next(
        (
            item for item in row.get("targets", ())
            if isinstance(item, Mapping) and str(item.get("backend", item.get("name", ""))) == backend
        ),
        {},
    )
    raw = row.get("raw_results", {}).get(backend, {}) if isinstance(row.get("raw_results"), Mapping) else {}
    return CausalSignature.build(
        program_ir=row.get("ccs_ir") or row.get("program_ir") or program,
        backend=backend,
        backend_version=str(target.get("version", row.get("environment", {}).get(backend, "")) or ""),
        dialect=str(target.get("dialect", "") or ""),
        adapter_version=str(target.get("adapter_version", "") or ""),
        lowering_version=str(target.get("lowering_version", "") or ""),
        normalized_difference={
            "kind": finding.get("kind", ""),
            "mismatch_class": finding.get("mismatch_class", ""),
            "signature": finding.get("signature", ""),
        },
        error_family={"error_type": raw.get("error_type", ""), "error": raw.get("error", "")},
        operation_sequence=operation_sequence,
        semantic_axes=finding.get("semantic_axes", ()) or (),
        reproduction_matrix=row.get("candidate_recheck", {}).get("matrix", {}) if isinstance(row.get("candidate_recheck"), Mapping) else {},
        minimized_case=minimized_case or case if isinstance(case, Mapping) else None,
    )


def _ir_prefix(value: Mapping[str, Any] | Sequence[Any] | None, prefix_length: int | None) -> Any:
    if isinstance(value, Mapping):
        operations = value.get("operations", value.get("nodes", ()))
        if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes, bytearray)):
            limit = len(operations) if prefix_length is None else max(0, int(prefix_length))
            return {"operations": list(operations)[:limit]}
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        limit = len(value) if prefix_length is None else max(0, int(prefix_length))
        return list(value)[:limit]
    return {}


def _matrix_items(value: Mapping[str, str] | Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        return sorted((str(key), str(item)) for key, item in value.items())
    return sorted((str(key), str(item)) for key, item in value)
