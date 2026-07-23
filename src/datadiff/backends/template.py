from __future__ import annotations

from datadiff.backends.base import Backend, BackendResult, PreparedTable
from datadiff.dsl import Program


class MinimalBackendTemplate(Backend):
    """Smallest P6 adapter shape; copy and replace execute_lowered only."""

    name = "template"
    session_reuse_policy = "stateless"

    def execute_lowered(
        self,
        tables: list[PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        return BackendResult(
            backend=self.name,
            status="missing",
            error_type="AdapterTemplateNotImplemented",
            error="implement execute_lowered for the new target",
        )
