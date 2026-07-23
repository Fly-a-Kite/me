from __future__ import annotations

from datadiff_osc.contract_engine import Observation
from datadiff_osc.contract_engine.mutations import (
    AXIS_FAULTS,
    MutationAuditResult,
    audit_axis_faults,
    audit_comparator_faults,
    audit_hyperedge_faults,
    inject_axis_fault,
)


def _observation(schema):
    return Observation.build(
        endpoint_id="left",
        status="ok",
        schema=schema,
        rows=[[1]],
        execution_metadata={"physical_layout": "contiguous"},
    )


def test_every_axis_fault_produces_a_valid_immutable_observation(int_schema):
    observation = _observation(int_schema)
    for axis in AXIS_FAULTS:
        mutated = inject_axis_fault(observation, axis)
        assert isinstance(mutated, Observation), axis
        assert mutated.digest, axis


def test_axis_audit_executes_detector_and_does_not_invent_kills(int_schema):
    seen = []
    report = audit_axis_faults(
        _observation(int_schema),
        lambda axis, mutated: seen.append((axis, mutated.digest)) or axis == "status",
        axes=("status", "numeric"),
    )
    assert tuple(item[0] for item in seen) == ("status", "numeric")
    assert report.killed_ids == ("status",)
    assert report.survived_ids == ("numeric",)


def test_comparator_and_hyperedge_helpers_execute_real_mutations(
    simple_compiled, int_schema
):
    observation = _observation(int_schema)
    comparator = audit_comparator_faults(
        simple_compiled.contract,
        observation,
        lambda mutant_id, contract, item: contract.digest != simple_compiled.contract.digest,
        mutant_ids=("ordered_to_bag", "logical_dtype_ignored"),
    )
    assert comparator.passed
    hyperedge = audit_hyperedge_faults(
        simple_compiled.contract,
        lambda mutant_id, contract: contract.digest != simple_compiled.contract.digest,
        mutant_ids=("delete_endpoint", "swap_target_control", "reverse_direction"),
    )
    assert hyperedge.passed


def test_mutation_audit_pass_requires_no_survivor():
    result = MutationAuditResult("x", ("a", "b"), ("a",), ("b",))
    assert result.kill_rate == 0.5
    assert result.passed is False

