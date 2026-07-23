from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from datadiff.backends.base import BackendResult
from datadiff.backends.spi import ADAPTER_SPI_VERSION
from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.targets import instantiate_target_backend, target_spec


ADAPTER_CONFORMANCE_SCHEMA_VERSION = "backend-adapter-conformance-v1"


@dataclass(frozen=True, slots=True)
class AdapterConformanceRow:
    backend: str
    passed: bool
    spi_version: str
    capability_digest: str
    execution_status: str
    stable_unsupported_skip_reason: str
    native_export_skip_reason: str
    checks: dict[str, bool]
    error_type: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def conformance_case() -> Case:
    return Case(
        case_id="p6-adapter-conformance",
        seed=0,
        tables=[
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("label", "str"),
                ],
                [
                    {"id": 1, "label": "alpha"},
                    {"id": 2, "label": None},
                ],
            )
        ],
        program=Program(
            "p6-adapter-conformance-program",
            0,
            [{"op": "select", "columns": ["id", "label"]}],
        ),
    )


def evaluate_adapter_conformance(backend_name: str) -> AdapterConformanceRow:
    spec = target_spec(backend_name)
    case = conformance_case()
    required = case_to_ccs_ir(case).required_capabilities
    checks: dict[str, bool] = {}
    try:
        backend = instantiate_target_backend(backend_name)
        manifest = backend.spi_manifest()
        checks["spi_version"] = manifest["schema_version"] == ADAPTER_SPI_VERSION
        checks["backend_identity"] = manifest["backend"] == backend_name
        checks["lowering_boundary"] = manifest["lowering"] == "prepare_tables+typed_program"
        checks["execution_boundary"] = manifest["execution"] == "execute_lowered"
        checks["normalization_boundary"] = manifest["normalization"] == "core_normalizer"
        checks["lifecycle_boundary"] = manifest["lifecycle"] == "reset_for_case+close"

        decision = spec.capability_decision(required_tokens=required)
        checks["declared_case_support"] = decision.supported
        lowered = backend.lower(case.tables, case.program)
        checks["lowered_tables_prepared"] = all(
            table.column_names for table in lowered.tables
        )
        result = backend.run(case.tables, case.program)
        checks["backend_result"] = isinstance(result, BackendResult)
        checks["execution_ok"] = result.status == "ok"

        unsupported = spec.capability_decision(
            required_tokens=("op:p6_deliberately_unsupported",)
        )
        expected_skip = (
            "unsupported_capability:operation:op:p6_deliberately_unsupported"
        )
        checks["stable_unsupported_skip"] = (
            not unsupported.supported and unsupported.skip_reason == expected_skip
        )

        native = backend.export_native(case.tables, case.program, format="p6_unknown")
        expected_native_skip = (
            "unsupported_capability:native_export_format:p6_unknown"
        )
        checks["stable_native_export_skip"] = (
            native.status == "unsupported"
            and native.skip_reason == expected_native_skip
        )
        backend.reset_for_case()
        backend.close()
        checks["lifecycle_calls"] = True
        return AdapterConformanceRow(
            backend=backend_name,
            passed=all(checks.values()),
            spi_version=str(manifest["schema_version"]),
            capability_digest=spec.capability_model.digest,
            execution_status=result.status,
            stable_unsupported_skip_reason=unsupported.skip_reason,
            native_export_skip_reason=native.skip_reason,
            checks=checks,
        )
    except Exception as exc:  # noqa: BLE001
        return AdapterConformanceRow(
            backend=backend_name,
            passed=False,
            spi_version=ADAPTER_SPI_VERSION,
            capability_digest=spec.capability_model.digest,
            execution_status="error",
            stable_unsupported_skip_reason="",
            native_export_skip_reason="",
            checks=checks,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def run_adapter_conformance(backends: Sequence[str]) -> dict[str, Any]:
    rows = [evaluate_adapter_conformance(backend) for backend in backends]
    passed = sum(1 for row in rows if row.passed)
    return {
        "schema_version": ADAPTER_CONFORMANCE_SCHEMA_VERSION,
        "backend_count": len(rows),
        "passed_count": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "rows": [row.to_dict() for row in rows],
    }
