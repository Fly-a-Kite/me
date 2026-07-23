from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from datadiff.canonicalization import short_canonical_hash
from datadiff.ccs_ir import ContractCarryingRelationalIR, case_to_ccs_ir
from datadiff.dsl import Case, Program
from datadiff.method_arms import IRMode


LEGACY_IR_SCHEMA_VERSION = "legacy-dict-ir-v1"


@dataclass(frozen=True, slots=True)
class ResolvedProgram:
    ir_mode: IRMode
    program: Program
    program_ir_digest: str
    ccs_ir: ContractCarryingRelationalIR | None = None
    resolution_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        if self.ccs_ir is not None:
            return {
                "ir_mode": self.ir_mode,
                "resolution_ms": self.resolution_ms,
                **self.ccs_ir.summary(),
            }
        return {
            "ir_mode": self.ir_mode,
            "schema_version": LEGACY_IR_SCHEMA_VERSION,
            "digest": self.program_ir_digest,
            "syntax_digest": self.program_ir_digest,
            "resolution_ms": self.resolution_ms,
            "node_count": len(self.program.operations),
            "source_relation_count": 0,
            "required_capabilities": [],
            "testing_objectives": [],
        }

    def semantic_plan_payload(self) -> dict[str, Any]:
        if self.ccs_ir is not None:
            return self.ccs_ir.semantic_payload()
        return self.program.to_dict()


def resolve_case_program(case: Case, config: Any) -> ResolvedProgram:
    started = time.perf_counter()
    ir_mode = _config_ir_mode(config)
    source_digest = _source_digest(case)
    cache_key = f"{ir_mode}:{source_digest}"
    cache = case._runtime_cache
    cached = cache.get(cache_key)
    if isinstance(cached, ResolvedProgram):
        return cached

    if ir_mode == "ccs_ir":
        ir = case_to_ccs_ir(case)
        resolved = ResolvedProgram(
            ir_mode=ir_mode,
            program=ir.to_program(),
            program_ir_digest=ir.digest,
            ccs_ir=ir,
            resolution_ms=(time.perf_counter() - started) * 1000.0,
        )
    else:
        digest = f"legacy-ir-{short_canonical_hash(case.program.to_dict(), 64)}"
        resolved = ResolvedProgram(
            ir_mode=ir_mode,
            program=case.program,
            program_ir_digest=digest,
            resolution_ms=(time.perf_counter() - started) * 1000.0,
        )
    cache.clear()
    cache[cache_key] = resolved
    return resolved


def _config_ir_mode(config: Any) -> IRMode:
    mode = str(config.method_policy.semantic.ir_mode)
    if mode not in {"legacy_dict", "ccs_ir"}:
        raise ValueError(f"unsupported IR mode: {mode}")
    return cast(IRMode, mode)


def _source_digest(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    return short_canonical_hash(
        {
            "tables": [table.to_dict() for table in case.tables],
            "program": case.program.to_dict(),
            "input_layouts": metadata.get("input_layouts", {}),
        },
        64,
    )
