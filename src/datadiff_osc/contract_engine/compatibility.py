from __future__ import annotations

import math
from typing import Any, Mapping

from datadiff_osc.contract_engine.compiler import (
    CompiledContract,
    ProgramSemantics,
    compile_hypercontract,
)
from datadiff_osc.contract_engine.domains import (
    AbstractState,
    LayoutDomain,
    ObservationDemand,
)
from datadiff_osc.contract_engine.model import EndpointRequirement
from datadiff_osc.contract_engine.rules import SemanticStep


V1_FACADE_SCHEMA_VERSION = "semantic-contract-lattice-v1"
V1_COMPARISON_VIEWS: tuple[str, ...] = (
    "exact",
    "ordered_value",
    "bag_value",
    "numeric_tolerant",
    "error_equivalent",
)


def v1_profile_relation(profile: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    view = str(profile.get("view", "bag_value"))
    if view == "exact":
        return "sequence_equal", {}
    if view == "ordered_value":
        return "sequence_equal", {}
    if view == "bag_value":
        return "bag_equal", {}
    if view == "error_equivalent":
        return "error_category_equal", {}
    if view == "numeric_tolerant":
        # v1 decimal rounding is intentionally not silently re-authorized.  The
        # caller must freeze explicit parameters in the compatibility corpus.
        explicit = profile.get("osc_numeric_parameters")
        if not isinstance(explicit, Mapping):
            raise ValueError(
                "numeric_tolerant v1 facade requires explicit OSC abs/rel/ULP parameters"
            )
        parameters = dict(explicit)
        if not ({"abs_tol", "rel_tol", "ulp_tol"} & set(parameters)):
            raise ValueError(
                "numeric_tolerant v1 facade requires an explicit non-rounding tolerance"
            )
        return "numeric_tolerant", parameters
    raise ValueError(f"unknown v1 comparison view: {view}")


def compile_v1_profile_facade(
    program: ProgramSemantics,
    profile: Mapping[str, Any],
    endpoint_requirements: tuple[EndpointRequirement, ...],
) -> CompiledContract:
    relation, parameters = v1_profile_relation(profile)
    view = str(profile.get("view", "bag_value"))
    if view == "numeric_tolerant":
        parameters = {**parameters, "collection": "bag"}
    return compile_hypercontract(
        program,
        endpoint_requirements=endpoint_requirements,
        relation_id=relation,
        relation_parameters=parameters,
        strength=f"v1-facade:{view}",
        final_demand=_v1_final_demand(view),
    )


def v1_facade_payload(compiled: CompiledContract) -> dict[str, Any]:
    state = compiled.forward.final_state
    view = (
        compiled.contract.strength.split(":", 1)[1]
        if compiled.contract.strength.startswith("v1-facade:")
        else "unknown"
    )
    return {
        "schema_version": V1_FACADE_SCHEMA_VERSION,
        "case_id": compiled.derivation.program_digest,
        "comparison_view": view,
        "precision_fixes": (
            ["explicit_abs_rel_ulp", "bag_bipartite_matching"]
            if view == "numeric_tolerant"
            else ["lossless_tagged_scalars"]
        ),
        "axes": {
            "ordering": {
                "axis": "ordering",
                "policy": "strict" if state.presentation_order.value in {"total", "partial"} else "canonicalized",
                "reason": "generated from OSC presentation-order compatibility view",
                "signals": [],
            },
            "null": _strict_v1_axis("null"),
            "nan": _strict_v1_axis("nan"),
            "dtype_coercion": _strict_v1_axis("dtype_coercion"),
            "error_equivalence": _strict_v1_axis("error_equivalence"),
            "layout_sensitivity": _strict_v1_axis("layout_sensitivity"),
            "determinism": _strict_v1_axis("determinism"),
        },
        "boundary_axes": [],
        "strict_axes": [
            "ordering", "null", "nan", "dtype_coercion",
            "error_equivalence", "layout_sensitivity", "determinism",
        ],
        "operation_contracts": [],
        "contract_tags": [f"osc-contract:{compiled.contract.digest}"],
        "osc_derivation_digest": compiled.derivation.digest,
    }


def program_semantics_from_v1_case(case: Any) -> ProgramSemantics:
    """Narrow bridge from the existing CCS-IR; no v1 verdict logic is imported."""

    from datadiff.ccs_ir import case_to_ccs_ir  # migration-only dependency

    ir = case_to_ccs_ir(case)
    primary = ir.source_relations[0]
    table = case.tables[0]
    values = [value for row in table.rows for value in row.values()]
    has_nulls = any(value is None for value in values)
    has_special = any(
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
        for value in values
    )
    layout = _layout_from_text(ir.input_layouts[0].representation)
    initial = AbstractState.initial(
        schema=tuple(
            (column.name, column.logical_type, column.nullable)
            for column in primary.columns
        ),
        has_nulls=has_nulls,
        has_special_floats=has_special,
        row_count=len(table.rows),
        layout=layout,
    )
    steps = tuple(
        SemanticStep.build(
            step_id=node.node_id,
            kind=node.kind,
            arguments=node.operation.arguments.to_dict(),
            expression_kinds=tuple(item.kind for item in node.scalar_expressions),
            aggregate_kinds=tuple(item.function for item in node.aggregates),
        )
        for node in ir.nodes
    )
    return ProgramSemantics(ir.digest, initial, steps)


def _layout_from_text(value: str) -> LayoutDomain:
    text = str(value).lower()
    for item in LayoutDomain:
        if item.value == text:
            return item
    if "slice" in text:
        return LayoutDomain.SLICED
    if "chunk" in text:
        return LayoutDomain.CHUNKED
    if "dictionary" in text:
        return LayoutDomain.DICTIONARY
    return LayoutDomain.UNKNOWN


def _strict_v1_axis(axis: str) -> dict[str, Any]:
    return {
        "axis": axis,
        "policy": "strict",
        "reason": "OSC compatibility facade does not grant global boundary permission",
        "signals": [],
    }


def _v1_final_demand(view: str) -> ObservationDemand:
    if view == "exact":
        return ObservationDemand(
            schema_names=True,
            schema_types=True,
            nullability=True,
            presentation_order=True,
            numeric=True,
            null_semantics=True,
        )
    if view in {"ordered_value", "bag_value", "numeric_tolerant"}:
        return ObservationDemand(
            schema_names=True,
            schema_types=False,
            nullability=False,
            presentation_order=view == "ordered_value",
            numeric=view == "numeric_tolerant",
            null_semantics=True,
        )
    if view == "error_equivalent":
        return ObservationDemand(
            schema_names=False,
            schema_types=False,
            nullability=False,
            cardinality=False,
            row_membership=False,
            duplicate_multiplicity=False,
            error_category=True,
        )
    raise ValueError(f"unknown v1 comparison view: {view}")
