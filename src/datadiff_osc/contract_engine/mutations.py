from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.model import (
    EndpointRequirement,
    HyperContract,
    Observation,
    RelationObligation,
)


COMPARATOR_WEAKENING_MUTANTS: tuple[str, ...] = (
    "nan_to_null",
    "negative_zero_to_positive_zero",
    "ordered_to_bag",
    "bag_to_set",
    "logical_dtype_ignored",
    "column_name_order_ignored",
    "error_category_collapsed",
    "timeout_crash_to_unsupported",
    "partial_order_freedom_globalized",
    "numeric_tolerance_expanded",
    "layout_relation_ignores_values",
    "presentation_evaluation_order_confused",
)

AXIS_FAULTS: tuple[str, ...] = (
    "status", "schema", "dtype", "nullability", "cardinality",
    "multiplicity", "order", "null", "nan", "numeric", "error",
    "layout_invariance", "determinism",
)

HYPEREDGE_MUTANTS: tuple[str, ...] = (
    "delete_endpoint", "swap_target_control", "reverse_direction",
    "drop_applicability_clause", "assume_invalid_symmetry",
    "tile_drop_a_only", "tile_drop_b_only",
)


@dataclass(frozen=True, slots=True)
class MutationAuditResult:
    category: str
    mutant_ids: tuple[str, ...]
    killed_ids: tuple[str, ...]
    survived_ids: tuple[str, ...]
    schema_version: str = "osc-contract-mutation-audit-v1"

    @property
    def kill_rate(self) -> float:
        return len(self.killed_ids) / len(self.mutant_ids) if self.mutant_ids else 1.0

    @property
    def passed(self) -> bool:
        return not self.survived_ids

    @property
    def digest(self) -> str:
        return stable_digest("osc-mutation-audit", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def mutate_contract(contract: HyperContract, mutant_id: str) -> HyperContract:
    obligations = list(contract.obligations)
    value_index = _value_obligation_index(obligations)
    if mutant_id in {"ordered_to_bag", "presentation_evaluation_order_confused"}:
        item = obligations[value_index]
        obligations[value_index] = replace(item, relation_id="bag_equal", strength="bag_exact_value")
    elif mutant_id == "bag_to_set":
        item = obligations[value_index]
        obligations[value_index] = replace(item, relation_id="set_equal_unique", strength="set_exact_value")
    elif mutant_id in {"logical_dtype_ignored", "column_name_order_ignored"}:
        obligations = [item for item in obligations if item.relation_id != "schema_equal"]
    elif mutant_id == "error_category_collapsed":
        item = obligations[value_index]
        obligations[value_index] = replace(item, relation_id="status_equal")
    elif mutant_id == "partial_order_freedom_globalized":
        item = obligations[value_index]
        obligations[value_index] = replace(item, relation_id="bag_equal", parameters=())
    elif mutant_id == "numeric_tolerance_expanded":
        item = obligations[value_index]
        obligations[value_index] = RelationObligation.build(
            obligation_id=item.obligation_id,
            relation_id="numeric_tolerant",
            endpoint_ids=item.endpoint_ids,
            components=item.components,
            parameters={"abs_tol": 1e9, "rel_tol": 1.0, "ulp_tol": 2**31},
            strength="numeric_unbounded_mutant",
            relation_properties=item.relation_properties,
        )
    elif mutant_id == "layout_relation_ignores_values":
        item = obligations[value_index]
        obligations[value_index] = replace(item, relation_id="status_equal", components=("status",))
    elif mutant_id == "delete_endpoint":
        if len(contract.endpoint_requirements) < 2:
            raise ValueError("delete_endpoint mutant requires at least two endpoints")
        removed = contract.endpoint_requirements[-1].endpoint_id
        requirements = contract.endpoint_requirements[:-1]
        obligations = [
            replace(item, endpoint_ids=tuple(value for value in item.endpoint_ids if value != removed))
            for item in obligations
            if len(item.endpoint_ids) > 1
        ]
        return replace(
            contract,
            endpoint_requirements=requirements,
            obligations=tuple(obligations),
            contract_id=contract.contract_id + "-mut-delete-endpoint",
        )
    elif mutant_id == "swap_target_control":
        if len(contract.endpoint_requirements) < 2:
            raise ValueError("swap mutant requires at least two endpoints")
        requirements = list(contract.endpoint_requirements)
        requirements[0], requirements[1] = requirements[1], requirements[0]
        obligations = [
            replace(item, endpoint_ids=tuple(reversed(item.endpoint_ids)))
            if len(item.endpoint_ids) == 2 else item
            for item in obligations
        ]
        return replace(
            contract,
            endpoint_requirements=tuple(requirements),
            obligations=tuple(obligations),
            contract_id=contract.contract_id + "-mut-swap",
        )
    elif mutant_id == "reverse_direction":
        item = obligations[value_index]
        obligations[value_index] = replace(
            item, endpoint_ids=tuple(reversed(item.endpoint_ids))
        )
    elif mutant_id == "drop_applicability_clause":
        return replace(
            contract,
            preconditions=contract.preconditions[1:],
            contract_id=contract.contract_id + "-mut-drop-pre",
        )
    elif mutant_id == "assume_invalid_symmetry":
        item = obligations[value_index]
        obligations[value_index] = replace(
            item,
            relation_properties=frozenset(
                (*item.relation_properties, "symmetric", "transitive")
            ),
        )
    elif mutant_id in {"tile_drop_a_only", "tile_drop_b_only"}:
        if len(contract.endpoint_requirements) != 4:
            raise ValueError("tile control mutant requires four endpoints")
        remove_index = 1 if mutant_id == "tile_drop_a_only" else 2
        removed = contract.endpoint_requirements[remove_index].endpoint_id
        requirements = tuple(
            item for index, item in enumerate(contract.endpoint_requirements)
            if index != remove_index
        )
        obligations = [
            replace(item, endpoint_ids=tuple(value for value in item.endpoint_ids if value != removed))
            for item in obligations
        ]
        return replace(
            contract,
            endpoint_requirements=requirements,
            obligations=tuple(obligations),
            contract_id=contract.contract_id + f"-mut-{mutant_id}",
        )
    elif mutant_id in {
        "nan_to_null", "negative_zero_to_positive_zero", "timeout_crash_to_unsupported"
    }:
        # These are observation-transformer mutants, not contract mutations.
        return replace(contract, contract_id=contract.contract_id + f"-mut-{mutant_id}")
    else:
        raise KeyError(mutant_id)
    return replace(
        contract,
        obligations=tuple(obligations),
        contract_id=contract.contract_id + f"-mut-{mutant_id}",
    )


def mutate_observation(observation: Observation, mutant_id: str) -> Observation:
    if mutant_id == "nan_to_null":
        rows = tuple(tuple(None if _is_nan(value) else value for value in row) for row in observation.rows)
        return Observation.build(
            endpoint_id=observation.endpoint_id,
            status=observation.status,
            schema=observation.schema,
            rows=rows,
            error_category=observation.error_category,
            error_type=observation.error_type,
            error_message=observation.error_message,
            execution_metadata=dict(observation.execution_metadata),
        )
    if mutant_id == "negative_zero_to_positive_zero":
        rows = tuple(tuple(0.0 if _is_negative_zero(value) else value for value in row) for row in observation.rows)
        return replace(observation, rows=rows)
    if mutant_id == "timeout_crash_to_unsupported" and observation.status in {"timeout", "crash"}:
        return Observation.build(
            endpoint_id=observation.endpoint_id,
            status="unsupported",
            schema=observation.schema,
            execution_metadata=dict(observation.execution_metadata),
        )
    raise KeyError(mutant_id)


def inject_axis_fault(observation: Observation, axis: str) -> Observation:
    if axis == "status":
        return Observation.build(
            endpoint_id=observation.endpoint_id,
            status="crash",
            schema=observation.schema,
            execution_metadata=dict(observation.execution_metadata),
        )
    if axis == "schema":
        return replace(observation, schema=tuple(reversed(observation.schema)))
    if axis == "dtype" and observation.schema:
        first = replace(observation.schema[0], logical_type="fault_type")
        return replace(observation, schema=(first, *observation.schema[1:]))
    if axis == "nullability" and observation.schema:
        first = replace(observation.schema[0], nullable=not observation.schema[0].nullable)
        return replace(observation, schema=(first, *observation.schema[1:]))
    if axis == "cardinality":
        fault_row = (
            observation.rows[-1]
            if observation.rows
            else tuple("fault" for _ in observation.schema)
        )
        return replace(observation, rows=(*observation.rows, fault_row))
    if axis == "multiplicity":
        fault_row = (
            observation.rows[0]
            if observation.rows
            else tuple("fault" for _ in observation.schema)
        )
        return replace(observation, rows=(*observation.rows, fault_row))
    if axis == "order":
        return replace(observation, rows=tuple(reversed(observation.rows)))
    if axis == "null" and observation.rows and observation.rows[0]:
        rows = list(observation.rows)
        rows[0] = (None, *rows[0][1:])
        return replace(observation, rows=tuple(rows))
    if axis == "nan" and observation.rows and observation.rows[0]:
        rows = list(observation.rows)
        rows[0] = (float("nan"), *rows[0][1:])
        return Observation.build(endpoint_id=observation.endpoint_id, status=observation.status, schema=observation.schema, rows=rows)
    if axis == "numeric" and observation.rows and observation.rows[0]:
        rows = list(observation.rows)
        value = rows[0][0]
        rows[0] = ((value + 1) if isinstance(value, (int, float)) else 1, *rows[0][1:])
        return replace(observation, rows=tuple(rows))
    if axis == "error":
        return Observation.build(
            endpoint_id=observation.endpoint_id,
            status="semantic_error",
            schema=observation.schema,
            error_category="fault_category",
            error_type="injected_fault",
            execution_metadata=dict(observation.execution_metadata),
        )
    if axis == "layout_invariance":
        metadata = dict(observation.execution_metadata)
        metadata["physical_layout"] = "fault_layout"
        rows = (
            *observation.rows,
            tuple("layout_value_fault" for _ in observation.schema),
        )
        return Observation.build(endpoint_id=observation.endpoint_id, status=observation.status, schema=observation.schema, rows=rows, execution_metadata=metadata)
    if axis == "determinism":
        metadata = dict(observation.execution_metadata)
        metadata["repeat_digest"] = "fault_repeat"
        return replace(observation, execution_metadata=tuple(sorted(metadata.items())))
    raise KeyError(axis)


def audit_mutants(
    category: str,
    mutant_ids: tuple[str, ...],
    killer: Callable[[str], bool],
) -> MutationAuditResult:
    killed = tuple(item for item in mutant_ids if bool(killer(item)))
    survived = tuple(item for item in mutant_ids if item not in set(killed))
    return MutationAuditResult(category, mutant_ids, killed, survived)


def audit_comparator_faults(
    contract: HyperContract,
    observation: Observation,
    killer: Callable[[str, HyperContract, Observation], bool],
    *,
    mutant_ids: tuple[str, ...] = COMPARATOR_WEAKENING_MUTANTS,
) -> MutationAuditResult:
    """Execute every comparator mutant against a caller-supplied witness."""

    observation_mutants = {
        "nan_to_null",
        "negative_zero_to_positive_zero",
        "timeout_crash_to_unsupported",
    }

    def execute(mutant_id: str) -> bool:
        mutated_contract = contract
        mutated_observation = observation
        if mutant_id in observation_mutants:
            try:
                mutated_observation = mutate_observation(observation, mutant_id)
            except KeyError:
                return False
        else:
            mutated_contract = mutate_contract(contract, mutant_id)
        return bool(killer(mutant_id, mutated_contract, mutated_observation))

    return audit_mutants("comparator", mutant_ids, execute)


def audit_axis_faults(
    observation: Observation,
    killer: Callable[[str, Observation], bool],
    *,
    axes: tuple[str, ...] = AXIS_FAULTS,
) -> MutationAuditResult:
    """Inject each frozen semantic axis fault and record real detections."""

    return audit_mutants(
        "axis",
        axes,
        lambda axis: bool(killer(axis, inject_axis_fault(observation, axis))),
    )


def audit_hyperedge_faults(
    contract: HyperContract,
    killer: Callable[[str, HyperContract], bool],
    *,
    mutant_ids: tuple[str, ...] = HYPEREDGE_MUTANTS,
) -> MutationAuditResult:
    """Mutate endpoint/hyperedge structure before invoking the detector."""

    def execute(mutant_id: str) -> bool:
        try:
            mutated = mutate_contract(contract, mutant_id)
        except ValueError:
            return False
        return bool(killer(mutant_id, mutated))

    return audit_mutants("hyperedge", mutant_ids, execute)


def _value_obligation_index(obligations: list[RelationObligation]) -> int:
    for index in range(len(obligations) - 1, -1, -1):
        if obligations[index].relation_id not in {"status_ok", "schema_equal", "cardinality_equal"}:
            return index
    if not obligations:
        raise ValueError("contract has no obligations")
    return len(obligations) - 1


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_negative_zero(value: Any) -> bool:
    return isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0
