#!/usr/bin/env python3
"""Measure staged-versus-exact parity; does not promote or launch anything."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json, stable_digest  # noqa: E402
from datadiff_osc.comparison.staged import StagedComparisonEngine  # noqa: E402
from datadiff_osc.contract_engine import (  # noqa: E402
    AbstractState,
    Endpoint,
    EndpointRequirement,
    Observation,
    ProgramSemantics,
    SchemaField,
    SemanticStep,
    compile_hypercontract,
    evaluate_applicability,
    execute_staged_comparison,
)
from datadiff_osc.contract_engine.domains import LayoutDomain  # noqa: E402
from datadiff_osc.runtime.benchmarks import audit_staged_exact_parity  # noqa: E402
from datadiff_osc.schemas import (  # noqa: E402
    ContractFingerprint,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    StagedComparisonRequest,
    StructuredExecutionOutcome,
)


@dataclass(frozen=True, slots=True)
class _Sample:
    sample_id: str
    observations: tuple[Observation, ...]
    execution_outcomes: tuple[StructuredExecutionOutcome, ...]


def _authority_fixture():
    capabilities = {"op:select", "type:int"}
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=endpoint_id,
            case_digest="synthetic-parity-corpus",
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="synthetic-adapter-v1",
            execution_mode="eager",
            physical_layout="contiguous",
            capabilities=capabilities,
        )
        for endpoint_id, backend in (("left", "backend-left"), ("right", "backend-right"))
    )
    program = ProgramSemantics(
        "osc-staged-parity-program",
        AbstractState.initial(
            schema=(("x", "int", False),),
            row_count=2,
            layout=LayoutDomain.CONTIGUOUS,
        ),
        (SemanticStep.build(step_id="select", kind="select"),),
    )
    requirements = tuple(
        EndpointRequirement(
            endpoint_id=item.endpoint_id,
            backend=item.backend,
            version_spec="*",
            required_capabilities=frozenset({"op:select"}),
        )
        for item in endpoints
    )
    compiled = compile_hypercontract(
        program,
        endpoint_requirements=requirements,
        relation_id="bag_equal",
    )
    applicability = evaluate_applicability(
        compiled.contract,
        endpoints,
        facts=frozenset(compiled.contract.preconditions),
        unresolved_rules=compiled.derivation.unresolved_facts,
    )
    return compiled.contract, applicability, endpoints


def _samples(groups: int, seed: int, endpoints):
    schema = (SchemaField("x", "int", False),)
    endpoint_digests = {item.endpoint_id: item.digest for item in endpoints}
    for index in range(groups):
        first = (seed + index * 17) % 997
        second = (seed + index * 31 + 1) % 997
        left = ((first,), (second,))
        right = left if index % 7 else ((first,), ((second + 1) % 997,))
        observations = (
            Observation.build(
                endpoint_id="left",
                status="ok",
                schema=schema,
                rows=left,
                execution_metadata={"endpoint_digest": endpoint_digests["left"]},
            ),
            Observation.build(
                endpoint_id="right",
                status="ok",
                schema=schema,
                rows=right,
                execution_metadata={"endpoint_digest": endpoint_digests["right"]},
            ),
        )
        execution_outcomes = (
            StructuredExecutionOutcome(
                endpoint_id="left",
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            ),
            StructuredExecutionOutcome(
                endpoint_id="right",
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            ),
        )
        yield _Sample(
            sample_id=f"synthetic-{seed}-{index:08d}",
            observations=observations,
            execution_outcomes=execution_outcomes,
        )


def _emit(payload, output: Path | None) -> None:
    text = canonical_json(payload) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=41000001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.groups < 1:
        parser.error("--groups must be positive")
    contract, applicability, endpoints = _authority_fixture()
    contract_fingerprint = ContractFingerprint(contract.digest, contract.registry_digest)

    def staged(sample: _Sample):
        request = StagedComparisonRequest(
            result_group_digest=stable_digest("osc-parity-result-group", sample.sample_id),
            contract_fingerprint=contract_fingerprint,
            endpoint_ids=("left", "right"),
            observer_ids=tuple(contract.observations),
            evidence_tier=EvidenceTier.SCREENING,
        )
        result, _ = StagedComparisonEngine.compare(
            request,
            contract,
            sample.observations,
            applicability,
            execution_outcomes=sample.execution_outcomes,
            endpoints=endpoints,
        )
        return result.verdict_kind, result.exact_escalated

    def exact(sample: _Sample):
        return execute_staged_comparison(
            contract,
            sample.observations,
            applicability,
            evidence_tier=EvidenceTier.AUDIT.value,
            endpoints=endpoints,
            force_exact=True,
            execution_outcomes=sample.execution_outcomes,
        ).verdict.kind

    report = audit_staged_exact_parity(
        _samples(args.groups, args.seed, endpoints),
        sample_id=lambda sample: sample.sample_id,
        staged=staged,
        exact=exact,
    )
    _emit(
        {
            "schema_version": "osc-staged-parity-script-output-v1",
            "authority": "datadiff_osc.contract_engine.execute_staged_comparison(force_exact=True)",
            "seed": args.seed,
            "contract_digest": contract.digest,
            "report": report,
            "phase6_gate_eligible": report.phase6_sample_floor_met and report.zero_discrepancy,
            "twenty_four_hour_run_authorized": False,
        },
        args.output,
    )
    return 0 if report.zero_discrepancy else 2


if __name__ == "__main__":
    raise SystemExit(main())
