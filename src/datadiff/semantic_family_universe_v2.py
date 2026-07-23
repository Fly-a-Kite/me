"""Versioned extension of the global semantic-family witness universe.

The v1 universe remains frozen in :mod:`datadiff.semantic_family_universe`.
This module adds one bounded, native target family for every production
backend and is consumed by the global-v3 witness portfolio.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from datadiff.semantic_family_universe import (
    SEMANTIC_FAMILY_PAIR_SCHEMA_VERSION,
    SemanticFamilyDefinition,
    confirmed_root_family_definitions,
    expansion_family_definitions,
    semantic_family_definitions,
)
from datadiff.targets import resolve_target_backends


SEMANTIC_FAMILY_UNIVERSE_V2_SCHEMA_VERSION = "semantic-family-universe-v2"
SEMANTIC_FAMILY_UNIVERSE_V2_ID = "family_universe_v2"
SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE = "latest_all_engines"
SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE = "goal_first_witness_global_v3"
SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID = "p8_semantic_witness_global_v3"


_EXPANSION_V2_FAMILIES: tuple[SemanticFamilyDefinition, ...] = (
    SemanticFamilyDefinition(
        family_id="pandas_nullable_string_normalization",
        mechanism_id="nullable_string_normalization_groupby",
        target_backend="pandas",
        control_backends=("pyarrow",),
        required_capabilities=(
            "op:mutate",
            "expr:string_strip",
            "expr:string_lower",
            "expr:string_upper",
            "expr:string_replace",
            "expr:string_null_if_empty",
            "op:coalesce",
            "op:fill_null",
            "op:groupby",
            "agg:count",
            "agg:nunique",
            "type:str",
            "nulls",
        ),
        semantic_dimensions=(
            "logical_type",
            "null_semantics",
            "operation",
            "cross_operation",
        ),
        source="coverage_expansion",
        root_id="target-family-pandas-nullable-string-normalization-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Exercise Pandas nullable-string transforms, empty-to-null conversion, "
            "fallback selection, and grouped observation."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="pyarrow_string_layout_predicate",
        mechanism_id="string_predicate_physical_layout",
        target_backend="pyarrow",
        control_backends=("pandas",),
        required_capabilities=(
            "op:mutate",
            "expr:string_strip",
            "expr:string_contains",
            "expr:string_starts_with",
            "expr:string_ends_with",
            "op:case_when",
            "op:groupby",
            "agg:count",
            "agg:nunique",
            "type:str",
            "nulls",
        ),
        semantic_dimensions=(
            "physical_layout",
            "operation",
            "null_semantics",
            "cross_operation",
        ),
        source="coverage_expansion",
        root_id="target-family-pyarrow-string-layout-predicate-v2",
        minimum_cell_count=18,
        priority=3,
        rationale=(
            "Run generic string predicates through contiguous, chunked, and "
            "dictionary Arrow input layouts with grouped observation."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="polars_arithmetic_cast_sortedness",
        mechanism_id="arithmetic_cast_sortedness_observation",
        target_backend="polars",
        control_backends=("polars_lazy",),
        required_capabilities=(
            "op:mutate",
            "expr:add_const",
            "expr:clip",
            "expr:abs",
            "expr:cast",
            "op:sort",
            "op:sortedness_check",
            "type:float",
            "nulls",
        ),
        semantic_dimensions=(
            "operation",
            "order_semantics",
            "execution_mode",
            "logical_type",
        ),
        source="coverage_expansion",
        root_id="target-family-polars-arithmetic-cast-sortedness-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Observe eager Polars arithmetic, clipping, casting, NULL ordering, "
            "and explicit sortedness checks against lazy execution."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="polars_lazy_case_string_membership",
        mechanism_id="lazy_string_case_membership_optimizer",
        target_backend="polars_lazy",
        control_backends=("polars",),
        required_capabilities=(
            "op:mutate",
            "expr:string_strip",
            "expr:string_lower",
            "expr:string_concat",
            "expr:string_slice",
            "expr:string_replace",
            "op:case_when",
            "op:semi_join",
            "op:anti_join",
            "op:sort",
            "type:str",
            "nulls",
        ),
        semantic_dimensions=(
            "plan_kind",
            "execution_mode",
            "null_semantics",
            "cross_operation",
        ),
        source="coverage_expansion",
        root_id="target-family-polars-lazy-case-string-membership-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Compose lazy string expression lowering, CASE classification, and "
            "semi/anti membership without relying on a known issue replay."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="duckdb_union_duplicate_global_aggregate",
        mechanism_id="union_duplicate_global_aggregate",
        target_backend="duckdb",
        control_backends=("pandas", "sqlite"),
        required_capabilities=(
            "op:union_all",
            "op:distinct",
            "op:aggregate",
            "agg:count",
            "agg:sum",
            "agg:mean",
            "agg:nunique",
            "nulls",
        ),
        semantic_dimensions=(
            "cross_operation",
            "null_semantics",
            "operation",
            "plan_kind",
        ),
        source="coverage_expansion",
        root_id="target-family-duckdb-union-duplicate-global-aggregate-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Exercise UNION ALL multiplicity, optional DISTINCT, and global "
            "aggregate behavior over duplicate and nullable inputs."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="sqlite_three_valued_membership",
        mechanism_id="three_valued_membership_tuple_absence",
        target_backend="sqlite",
        control_backends=("pandas", "duckdb"),
        required_capabilities=(
            "op:semi_join",
            "op:anti_join",
            "op:tuple_absence_filter",
            "op:case_when",
            "op:groupby",
            "agg:count",
            "nulls",
        ),
        semantic_dimensions=(
            "null_semantics",
            "logical_type",
            "cross_operation",
            "plan_kind",
        ),
        source="coverage_expansion",
        root_id="target-family-sqlite-three-valued-membership-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Compare single- and multi-key semi/anti membership with native SQL "
            "row-value NOT IN behavior at left- and right-NULL boundaries."
        ),
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_null_setop_aggregate",
        mechanism_id="null_setop_grouped_aggregate",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=(
            "op:union_all",
            "op:drop_nulls",
            "op:fill_null",
            "op:coalesce",
            "op:groupby",
            "agg:sum",
            "agg:min",
            "agg:mean",
            "agg:count",
            "nulls",
        ),
        semantic_dimensions=(
            "null_semantics",
            "plan_kind",
            "cross_operation",
            "operation",
        ),
        source="coverage_expansion",
        root_id="target-family-datafusion-null-setop-aggregate-v2",
        minimum_cell_count=12,
        priority=3,
        rationale=(
            "Compose UNION ALL with three NULL-handling policies and grouped exact "
            "or statistical aggregates in DataFusion."
        ),
    ),
)


_FAMILIES_V2 = (*semantic_family_definitions(), *_EXPANSION_V2_FAMILIES)


def expansion_v2_family_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _EXPANSION_V2_FAMILIES


def semantic_family_v2_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _FAMILIES_V2


def semantic_family_v2_definition(family_id: str) -> SemanticFamilyDefinition:
    matches = [item for item in _FAMILIES_V2 if item.family_id == str(family_id)]
    if len(matches) != 1:
        raise KeyError(family_id)
    return matches[0]


def semantic_family_universe_v2_manifest() -> dict[str, Any]:
    backends = tuple(
        resolve_target_backends(target_suite=SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE)
    )
    families = [item.manifest() for item in _FAMILIES_V2]
    backend_pairs = [
        {
            "schema_version": SEMANTIC_FAMILY_PAIR_SCHEMA_VERSION,
            "family_id": family.family_id,
            "mechanism_id": family.mechanism_id,
            "backend": backend,
            "role": family.backend_role(backend),
            "required": family.backend_role(backend) != "out_of_scope",
        }
        for family in _FAMILIES_V2
        for backend in backends
    ]
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_V2_SCHEMA_VERSION,
        "universe_id": SEMANTIC_FAMILY_UNIVERSE_V2_ID,
        "inherits_universe_id": "family_universe_v1",
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE,
        "backends": list(backends),
        "global_v3_generation_mode": SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
        "global_v3_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
        "families": families,
        "backend_pairs": backend_pairs,
        "summary": {
            "backend_count": len(backends),
            "family_count": len(_FAMILIES_V2),
            "inherited_family_count": len(semantic_family_definitions()),
            "confirmed_root_family_count": len(confirmed_root_family_definitions()),
            "coverage_expansion_family_count": (
                len(expansion_family_definitions()) + len(_EXPANSION_V2_FAMILIES)
            ),
            "new_coverage_expansion_family_count": len(_EXPANSION_V2_FAMILIES),
            "backend_pair_count": len(backend_pairs),
            "required_backend_pair_count": sum(
                bool(item["required"]) for item in backend_pairs
            ),
        },
    }
    payload["digest"] = _digest(payload)
    return payload


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "semantic-family-universe-" + hashlib.sha256(encoded).hexdigest()
