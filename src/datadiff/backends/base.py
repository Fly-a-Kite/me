from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import ColumnSpec, Program, TableData
from datadiff.physical_plan import PhysicalPlanBundle


@dataclass(slots=True)
class PreparedTable:
    name: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Any]]
    column_names: tuple[str, ...]
    row_tuples: tuple[tuple[Any, ...], ...]
    columns_data: dict[str, list[Any]]


def prepare_table(table: TableData | PreparedTable) -> PreparedTable:
    if isinstance(table, PreparedTable):
        return table
    column_names = tuple(column.name for column in table.columns)
    row_tuples = tuple(
        tuple(row.get(column_name) for column_name in column_names)
        for row in table.rows
    )
    columns_data = {
        column_name: [row_values[index] for row_values in row_tuples]
        for index, column_name in enumerate(column_names)
    }
    return PreparedTable(
        name=table.name,
        columns=list(table.columns),
        rows=table.rows,
        column_names=column_names,
        row_tuples=row_tuples,
        columns_data=columns_data,
    )


def prepare_tables(tables: Sequence[TableData | PreparedTable]) -> list[PreparedTable]:
    return [prepare_table(table) for table in tables]


@dataclass(slots=True)
class BackendResult:
    backend: str
    status: str  # ok / error / timeout / missing
    data: Any = None
    error_type: str = ""
    error: str = ""
    duration_ms: float = 0.0
    physical_plan: PhysicalPlanBundle | None = None
    capability_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "status": self.status,
            "data": self.data,
            "error_type": self.error_type,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }
        if self.physical_plan is not None:
            payload["physical_plan"] = self.physical_plan.to_dict()
        if self.capability_decision is not None:
            payload["capability_decision"] = self.capability_decision
        return payload

    def summary_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "status": self.status,
            "error_type": self.error_type,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }
        if self.physical_plan is not None:
            payload["physical_plan"] = self.physical_plan.to_dict()
        if self.capability_decision is not None:
            payload["capability_decision"] = self.capability_decision
        return payload


@dataclass(slots=True)
class NativeRows:
    columns: list[str]
    row_values: list[list[Any]]
    column_types: list[str] | None = None

    def rows(self, named: bool = False):
        if named:
            return [dict(zip(self.columns, row)) for row in self.row_values]
        return self.row_values


class Backend:
    name = "base"
    session_reuse_policy = "stateless"
    # Most stateful adapters rely on the session executor to invoke
    # ``reset_for_case``.  An adapter that necessarily resets inside its own
    # execution entry point can opt out of a duplicate executor-side reset.
    session_reset_managed_by_backend = False
    physical_plan_collection_mode = "disabled"
    plan_collection_support = "unsupported"
    native_export_support = "unsupported"
    result_contract = "backend-result-v1"
    input_physical_layout = "default"
    supported_physical_layouts: tuple[str, ...] = ()

    @property
    def physical_plan_collection_enabled(self) -> bool:
        return self.physical_plan_collection_mode != "disabled"

    def configure_physical_plan_collection(self, mode: str | bool) -> None:
        normalized = "full" if mode is True else "disabled" if mode is False else str(mode)
        if normalized not in {"disabled", "fingerprint", "full"}:
            raise ValueError(f"unsupported physical plan collection mode: {normalized}")
        self.physical_plan_collection_mode = normalized

    def configure_physical_layout(self, layout: str) -> None:
        self.input_physical_layout = str(layout or "default")

    def reset_for_case(self) -> None:
        """Reset mutable adapter state before a reused-session execution."""

        return None

    def close(self) -> None:
        """Release adapter-owned lifecycle state."""

        return None

    def lower(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
    ) -> "LoweredCase":
        from datadiff.backends.spi import LoweredCase

        return LoweredCase(
            tables=tuple(prepare_tables(tables)),
            program=program,
            lowering_metadata={
                "schema_version": "backend-lowering-v1",
                "backend": self.name,
            },
        )

    def execute_lowered(
        self,
        tables: list[PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        raise NotImplementedError

    def run(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        from datadiff.backends.spi import validate_backend_result

        lowered = self.lower(tables, program)
        result = self.execute_lowered(
            list(lowered.tables),
            lowered.program,
            timeout_s=timeout_s,
        )
        validate_backend_result(result, backend_name=self.name)
        return result

    def export_native(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        *,
        format: str,
    ) -> "NativeExportResult":
        from datadiff.backends.spi import NativeExportResult

        return NativeExportResult(
            status="unsupported",
            format=str(format),
            skip_reason=f"unsupported_capability:native_export_format:{format}",
        )

    def spi_manifest(self) -> dict[str, Any]:
        from datadiff.backends.spi import AdapterSPIManifest

        return AdapterSPIManifest(
            backend=self.name,
            session_reuse_policy=self.session_reuse_policy,
            lowering="prepare_tables+typed_program",
            execution="execute_lowered",
            normalization="core_normalizer",
            plan_collection=self.plan_collection_support,
            native_export=self.native_export_support,
            lifecycle="reset_for_case+close",
            result_contract=self.result_contract,
        ).to_dict()
