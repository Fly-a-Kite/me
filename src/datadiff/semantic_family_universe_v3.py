"""Issue-risk-class extension of the semantic-family witness universe.

The upstream issue corpus is used to identify recurring *classes* of failures,
not to embed issue-specific reproducers in the runtime.  This module is the
declarative layer for Global-v4: it freezes family identities, backend scopes,
pipeline evidence, and the bounded axis products consumed by the registry and
audits.  Builders and backend execution remain separate layers.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from datadiff.semantic_family_universe_v2 import (
    expansion_v2_family_definitions,
    semantic_family_v2_definitions,
)
from datadiff.targets import resolve_target_backends


SEMANTIC_FAMILY_UNIVERSE_V3_SCHEMA_VERSION = "semantic-family-universe-v3"
SEMANTIC_FAMILY_UNIVERSE_V3_ID = "family_universe_v3"
SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE = "latest_all_engines"
SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE = "goal_first_witness_global_v4"
SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID = "p8_semantic_witness_global_v4"

CROSS_BACKEND_RISK_FAMILY_ID = "cross_backend_issue_risk_pipeline"
EXACT_DTYPE_FAMILY_ID = "cross_backend_exact_dtype_pipeline"
PYARROW_LAYOUT_RISK_FAMILY_ID = "pyarrow_layout_interaction_pipeline"

DATA_PATTERNS: tuple[str, ...] = ("baseline", "reordered_duplicate")
PYARROW_LAYOUTS: tuple[str, ...] = (
    "contiguous",
    "sliced",
    "chunked",
    "dictionary",
)

CROSS_BACKEND_PIPELINES: tuple[str, ...] = (
    "outer_join_coalesce_distinct_topk",
    "date_cast_union_running_sum_topk",
    "string_token_transform_join_window",
    "string_contains_anti_join_offset",
    "utf8_slice_length_groupby",
    "post_groupby_join_global_aggregate",
    "numeric_clip_division_anti_window",
    "drop_nulls_coalesce_distinct_join_topk",
    "bool_not_union_distinct_aggregate",
    "path_basename_keyed_pick",
    "polars_reverse_division_columns",
    "sortedness_null_placement",
    "tuple_absence_filter",
    "bool_null_groupby_agg",
    "aggregate_mean_matrix",
    "prefix_suffix_bool_membership",
    "filter_null_agg_topk",
    "scalar_abs_upper_filter",
)

EXACT_DTYPE_PIPELINES: tuple[str, ...] = (
    "outer_join_coalesce_distinct_topk",
    "date_cast_union_running_sum_topk",
    "string_token_transform_join_window",
    "utf8_slice_length_groupby",
    "bool_not_union_distinct_aggregate",
)

PYARROW_LAYOUT_PIPELINES: tuple[str, ...] = (
    "utf8_slice_length_groupby",
    "bool_not_union_distinct_aggregate",
    "date_cast_union_running_sum_topk",
)

GENERIC_OPERATION_KINDS: tuple[str, ...] = (
    "aggregate",
    "anti_join",
    "case_when",
    "coalesce",
    "distinct",
    "drop_nulls",
    "fill_null",
    "filter",
    "groupby",
    "join",
    "limit",
    "mutate",
    "offset",
    "row_number_filter",
    "running_sum",
    "select",
    "semi_join",
    "sort",
    "sortedness_check",
    "tuple_absence_filter",
    "union_all",
)

GENERIC_EXPRESSION_KINDS: tuple[str, ...] = (
    "abs",
    "add_const",
    "arith_const",
    "bool_not",
    "cast",
    "clip",
    "date_part",
    "reverse_division_columns",
    "string_basename",
    "string_concat",
    "string_contains",
    "string_ends_with",
    "string_length",
    "string_lower",
    "string_null_if_empty",
    "string_replace",
    "string_slice",
    "string_split_part",
    "string_starts_with",
    "string_strip",
    "string_upper",
)

GENERIC_AGGREGATE_FUNCTIONS: tuple[str, ...] = (
    "all",
    "any",
    "count",
    "max",
    "mean",
    "min",
    "nunique",
    "sum",
)

BOUNDARY_PROFILE_IDS_BY_RISK_CLASS: dict[str, tuple[str, ...]] = {
    "null_cardinality_interaction": (
        "duplicate_heavy_keys",
        "null_truth_table",
    ),
    "optimizer_rewrite_boundary": (
        "empty_singleton_transition",
        "null_truth_table",
    ),
    "string_unicode_membership": (
        "null_truth_table",
        "duplicate_heavy_keys",
        "timezone_transition_strings",
    ),
    "aggregate_observation": (
        "signed_zero_nan_infinity",
        "null_truth_table",
        "empty_singleton_transition",
    ),
    "dtype_numeric_expression": (
        "signed_zero_nan_infinity",
        "decimal_scale_edges",
        "null_truth_table",
    ),
    "projection_layout_schema": (
        "duplicate_heavy_keys",
        "null_truth_table",
    ),
    "ordering_window_topk": (
        "signed_zero_nan_infinity",
        "null_truth_table",
    ),
}


@dataclass(frozen=True, slots=True)
class PipelineEvidenceSpec:
    pipeline_id: str
    risk_class: str
    required_operations: tuple[str, ...]
    required_expressions: tuple[str, ...] = ()
    required_aggregates: tuple[str, ...] = ()
    aggregate_any_of: tuple[str, ...] = ()
    required_operation_chains: tuple[tuple[str, ...], ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "risk_class": self.risk_class,
            "required_operations": list(self.required_operations),
            "required_expressions": list(self.required_expressions),
            "required_aggregates": list(self.required_aggregates),
            "aggregate_any_of": list(self.aggregate_any_of),
            "required_operation_chains": [
                list(chain) for chain in self.required_operation_chains
            ],
        }


@dataclass(frozen=True, slots=True)
class SemanticFamilyV3WitnessSpec:
    family_id: str
    pipelines: tuple[str, ...]
    data_patterns: tuple[str, ...]
    execution_backends: tuple[str, ...]
    layouts: tuple[str, ...] = ()
    comparison_view: str = ""
    boundary_data_patterns: tuple[str, ...] = ()
    backend_scope_policy: str = "full"
    screening_plan_policy: str = "finding_only"
    cache_reuse_source_family_id: str = ""
    cache_reuse_axes: tuple[str, ...] = ()

    @property
    def axes(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        axes: list[tuple[str, tuple[str, ...]]] = [
            ("pipeline", self.pipelines),
        ]
        if self.layouts:
            axes.append(("layout", self.layouts))
        axes.append(("data_pattern", self.data_patterns))
        return tuple(axes)

    @property
    def cell_count(self) -> int:
        count = 1
        for _name, values in self.axes:
            count *= len(values)
        return count

    def manifest(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "execution_backends": list(self.execution_backends),
            "axes": [
                {"name": name, "values": list(values)}
                for name, values in self.axes
            ],
            "cell_count": self.cell_count,
            "comparison_view": self.comparison_view,
            "boundary_data_patterns": list(self.boundary_data_patterns),
            "backend_scope_policy": self.backend_scope_policy,
            "screening_plan_policy": self.screening_plan_policy,
            "cache_reuse_source_family_id": self.cache_reuse_source_family_id,
            "cache_reuse_axes": list(self.cache_reuse_axes),
        }


_PIPELINE_EVIDENCE: tuple[PipelineEvidenceSpec, ...] = (
    PipelineEvidenceSpec(
        "outer_join_coalesce_distinct_topk",
        "null_cardinality_interaction",
        ("join", "coalesce", "distinct", "groupby"),
        ("string_lower", "string_null_if_empty", "string_strip"),
        ("count", "max", "sum"),
        required_operation_chains=(("join", "distinct"),),
    ),
    PipelineEvidenceSpec(
        "date_cast_union_running_sum_topk",
        "optimizer_rewrite_boundary",
        ("union_all", "filter", "distinct", "running_sum", "row_number_filter"),
        ("cast", "date_part"),
        required_operation_chains=(
            ("union_all", "distinct", "filter"),
            ("filter", "running_sum", "row_number_filter"),
        ),
    ),
    PipelineEvidenceSpec(
        "string_token_transform_join_window",
        "string_unicode_membership",
        ("semi_join", "distinct", "running_sum"),
        (
            "string_concat",
            "string_replace",
            "string_slice",
            "string_split_part",
            "string_strip",
        ),
        required_operation_chains=(("semi_join", "distinct", "running_sum"),),
    ),
    PipelineEvidenceSpec(
        "string_contains_anti_join_offset",
        "string_unicode_membership",
        ("anti_join", "case_when", "offset", "limit"),
        ("string_contains", "string_null_if_empty", "string_strip"),
        required_operation_chains=(("anti_join", "offset", "limit"),),
    ),
    PipelineEvidenceSpec(
        "utf8_slice_length_groupby",
        "string_unicode_membership",
        ("mutate", "groupby"),
        ("string_length", "string_slice"),
        ("count", "nunique"),
        required_operation_chains=(("mutate", "groupby"),),
    ),
    PipelineEvidenceSpec(
        "post_groupby_join_global_aggregate",
        "aggregate_observation",
        ("groupby", "filter", "join", "aggregate"),
        required_aggregates=("count", "sum"),
        required_operation_chains=(("groupby", "join", "aggregate"),),
    ),
    PipelineEvidenceSpec(
        "numeric_clip_division_anti_window",
        "dtype_numeric_expression",
        ("anti_join", "running_sum", "case_when"),
        ("arith_const", "clip"),
        required_operation_chains=(("anti_join", "running_sum"),),
    ),
    PipelineEvidenceSpec(
        "drop_nulls_coalesce_distinct_join_topk",
        "projection_layout_schema",
        ("coalesce", "drop_nulls", "distinct", "join", "select"),
        required_operation_chains=(("join", "select", "drop_nulls"),),
    ),
    PipelineEvidenceSpec(
        "bool_not_union_distinct_aggregate",
        "null_cardinality_interaction",
        ("union_all", "distinct", "row_number_filter", "groupby"),
        ("bool_not",),
        ("count", "sum"),
        required_operation_chains=(("row_number_filter", "groupby"),),
    ),
    PipelineEvidenceSpec(
        "path_basename_keyed_pick",
        "projection_layout_schema",
        ("mutate", "row_number_filter", "select"),
        ("string_basename",),
        required_operation_chains=(("row_number_filter", "select"),),
    ),
    PipelineEvidenceSpec(
        "polars_reverse_division_columns",
        "dtype_numeric_expression",
        ("mutate", "select"),
        ("reverse_division_columns",),
        required_operation_chains=(("mutate", "select"),),
    ),
    PipelineEvidenceSpec(
        "sortedness_null_placement",
        "ordering_window_topk",
        ("sort", "sortedness_check"),
        required_operation_chains=(("sort", "sortedness_check"),),
    ),
    PipelineEvidenceSpec(
        "tuple_absence_filter",
        "null_cardinality_interaction",
        ("tuple_absence_filter", "select", "sort"),
        required_operation_chains=(("tuple_absence_filter", "select", "sort"),),
    ),
    PipelineEvidenceSpec(
        "bool_null_groupby_agg",
        "aggregate_observation",
        ("groupby", "sort"),
        required_aggregates=("all", "any", "count", "max", "min", "nunique", "sum"),
        required_operation_chains=(("groupby", "sort"),),
    ),
    PipelineEvidenceSpec(
        "aggregate_mean_matrix",
        "aggregate_observation",
        ("groupby", "sort"),
        required_aggregates=("count", "mean", "min"),
        required_operation_chains=(("groupby", "sort"),),
    ),
    PipelineEvidenceSpec(
        "prefix_suffix_bool_membership",
        "string_unicode_membership",
        ("anti_join", "row_number_filter", "groupby"),
        ("string_starts_with", "string_ends_with", "string_strip"),
        ("sum",),
        required_operation_chains=(("anti_join", "row_number_filter", "groupby"),),
    ),
    PipelineEvidenceSpec(
        "filter_null_agg_topk",
        "optimizer_rewrite_boundary",
        ("filter", "mutate", "groupby", "limit"),
        ("add_const",),
        aggregate_any_of=("min", "max"),
        required_operation_chains=(("filter", "groupby", "limit"),),
    ),
    PipelineEvidenceSpec(
        "scalar_abs_upper_filter",
        "dtype_numeric_expression",
        ("fill_null", "mutate", "filter", "sort", "select"),
        ("abs", "add_const", "string_upper"),
        required_operation_chains=(("mutate", "filter", "sort", "select"),),
    ),
)

_PIPELINE_EVIDENCE_BY_ID = {
    item.pipeline_id: item for item in _PIPELINE_EVIDENCE
}

_WITNESS_SPECS: tuple[SemanticFamilyV3WitnessSpec, ...] = (
    SemanticFamilyV3WitnessSpec(
        family_id=CROSS_BACKEND_RISK_FAMILY_ID,
        pipelines=CROSS_BACKEND_PIPELINES,
        data_patterns=DATA_PATTERNS,
        execution_backends=(
            "pandas",
            "pyarrow",
            "polars",
            "polars_lazy",
            "duckdb",
            "sqlite",
            "datafusion",
        ),
        boundary_data_patterns=("reordered_duplicate",),
        backend_scope_policy="balanced_axis_coverage",
        screening_plan_policy="finding_only",
    ),
    SemanticFamilyV3WitnessSpec(
        family_id=EXACT_DTYPE_FAMILY_ID,
        pipelines=EXACT_DTYPE_PIPELINES,
        data_patterns=DATA_PATTERNS,
        execution_backends=(
            "pandas",
            "pyarrow",
            "polars",
            "polars_lazy",
            "datafusion",
        ),
        comparison_view="exact",
        screening_plan_policy="finding_only",
        cache_reuse_source_family_id=CROSS_BACKEND_RISK_FAMILY_ID,
        cache_reuse_axes=("pipeline", "data_pattern"),
    ),
    SemanticFamilyV3WitnessSpec(
        family_id=PYARROW_LAYOUT_RISK_FAMILY_ID,
        pipelines=PYARROW_LAYOUT_PIPELINES,
        layouts=PYARROW_LAYOUTS,
        data_patterns=DATA_PATTERNS,
        execution_backends=("pandas", "pyarrow"),
        screening_plan_policy="finding_only",
    ),
)

_WITNESS_SPEC_BY_FAMILY = {item.family_id: item for item in _WITNESS_SPECS}

_COMMON_PIPELINE_CAPABILITIES = tuple(
    [f"op:{item}" for item in GENERIC_OPERATION_KINDS]
    + [f"expr:{item}" for item in GENERIC_EXPRESSION_KINDS]
    + [f"agg:{item}" for item in GENERIC_AGGREGATE_FUNCTIONS]
    + ["type:bool", "type:float", "type:int", "type:str", "nulls"]
)

_EXPANSION_V3_FAMILIES: tuple[SemanticFamilyDefinition, ...] = (
    SemanticFamilyDefinition(
        family_id=CROSS_BACKEND_RISK_FAMILY_ID,
        mechanism_id="issue_risk_class_cross_backend_pipeline",
        target_backend="pandas",
        control_backends=(
            "pyarrow",
            "polars",
            "polars_lazy",
            "duckdb",
            "sqlite",
            "datafusion",
        ),
        required_capabilities=_COMMON_PIPELINE_CAPABILITIES,
        semantic_dimensions=(
            "operation",
            "cross_operation",
            "null_semantics",
            "order_semantics",
            "logical_type",
            "plan_kind",
            "execution_mode",
        ),
        source="coverage_expansion",
        root_id="target-family-cross-backend-issue-risk-pipeline-v3",
        minimum_cell_count=36,
        priority=4,
        rationale=(
            "Abstract recurring issue classes into generic multi-operation pipelines "
            "and execute every frozen pipeline on all production backends."
        ),
    ),
    SemanticFamilyDefinition(
        family_id=EXACT_DTYPE_FAMILY_ID,
        mechanism_id="lossless_value_and_logical_dtype_observation",
        target_backend="datafusion",
        control_backends=("pandas", "pyarrow", "polars", "polars_lazy"),
        required_capabilities=(
            "op:union_all",
            "op:join",
            "op:groupby",
            "op:mutate",
            "op:distinct",
            "expr:cast",
            "expr:bool_not",
            "expr:string_slice",
            "agg:count",
            "agg:sum",
            "type:bool",
            "type:int",
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
        root_id="target-family-cross-backend-exact-dtype-pipeline-v3",
        minimum_cell_count=10,
        priority=4,
        rationale=(
            "Retain logical dtype together with exact semantic values across the "
            "five dataframe/Arrow execution engines that share the contract."
        ),
    ),
    SemanticFamilyDefinition(
        family_id=PYARROW_LAYOUT_RISK_FAMILY_ID,
        mechanism_id="physical_layout_operation_interaction",
        target_backend="pyarrow",
        control_backends=("pandas",),
        required_capabilities=(
            "op:union_all",
            "op:mutate",
            "op:distinct",
            "op:groupby",
            "op:row_number_filter",
            "expr:cast",
            "expr:bool_not",
            "expr:string_length",
            "expr:string_slice",
            "agg:count",
            "agg:nunique",
            "agg:sum",
            "type:bool",
            "type:int",
            "type:str",
            "nulls",
        ),
        semantic_dimensions=(
            "physical_layout",
            "logical_type",
            "operation",
            "cross_operation",
        ),
        source="coverage_expansion",
        root_id="target-family-pyarrow-layout-interaction-pipeline-v3",
        minimum_cell_count=24,
        priority=4,
        rationale=(
            "Cross contiguous, sliced, chunked, and dictionary layouts with generic "
            "string, boolean, cast, set, window, and aggregate pipelines."
        ),
    ),
)

_FAMILIES_V3 = (*semantic_family_v2_definitions(), *_EXPANSION_V3_FAMILIES)


def expansion_v3_family_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _EXPANSION_V3_FAMILIES


def semantic_family_v3_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _FAMILIES_V3


def semantic_family_v3_definition(family_id: str) -> SemanticFamilyDefinition:
    matches = [item for item in _FAMILIES_V3 if item.family_id == str(family_id)]
    if len(matches) != 1:
        raise KeyError(family_id)
    return matches[0]


def semantic_family_v3_witness_specs() -> tuple[SemanticFamilyV3WitnessSpec, ...]:
    return _WITNESS_SPECS


def semantic_family_v3_witness_spec(family_id: str) -> SemanticFamilyV3WitnessSpec:
    return _WITNESS_SPEC_BY_FAMILY[str(family_id)]


def pipeline_evidence_specs() -> tuple[PipelineEvidenceSpec, ...]:
    return _PIPELINE_EVIDENCE


def pipeline_evidence_spec(pipeline_id: str) -> PipelineEvidenceSpec:
    return _PIPELINE_EVIDENCE_BY_ID[str(pipeline_id)]


def issue_risk_boundary_profile_ids(risk_class: str) -> tuple[str, ...]:
    return BOUNDARY_PROFILE_IDS_BY_RISK_CLASS[str(risk_class)]


def semantic_family_universe_v3_manifest() -> dict[str, Any]:
    backends = tuple(
        resolve_target_backends(target_suite=SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE)
    )
    families = [item.manifest() for item in _FAMILIES_V3]
    backend_pairs = [
        {
            "schema_version": SEMANTIC_FAMILY_PAIR_SCHEMA_VERSION,
            "family_id": family.family_id,
            "mechanism_id": family.mechanism_id,
            "backend": backend,
            "role": family.backend_role(backend),
            "required": family.backend_role(backend) != "out_of_scope",
        }
        for family in _FAMILIES_V3
        for backend in backends
    ]
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_V3_SCHEMA_VERSION,
        "universe_id": SEMANTIC_FAMILY_UNIVERSE_V3_ID,
        "inherits_universe_id": "family_universe_v2",
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE,
        "backends": list(backends),
        "global_v4_generation_mode": SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
        "global_v4_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID,
        "families": families,
        "witness_specs": [item.manifest() for item in _WITNESS_SPECS],
        "pipeline_evidence": [item.manifest() for item in _PIPELINE_EVIDENCE],
        "backend_pairs": backend_pairs,
        "summary": {
            "backend_count": len(backends),
            "family_count": len(_FAMILIES_V3),
            "inherited_family_count": len(semantic_family_v2_definitions()),
            "confirmed_root_family_count": len(confirmed_root_family_definitions()),
            "coverage_expansion_family_count": (
                len(expansion_family_definitions())
                + len(expansion_v2_family_definitions())
                + len(_EXPANSION_V3_FAMILIES)
            ),
            "new_coverage_expansion_family_count": len(_EXPANSION_V3_FAMILIES),
            "new_witness_cell_count": sum(item.cell_count for item in _WITNESS_SPECS),
            "issue_risk_class_count": len(
                {item.risk_class for item in _PIPELINE_EVIDENCE}
            ),
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


if {item.pipeline_id for item in _PIPELINE_EVIDENCE} != set(CROSS_BACKEND_PIPELINES):
    raise AssertionError("pipeline evidence must exactly cover the frozen risk palette")
if {item.family_id for item in _WITNESS_SPECS} != {
    item.family_id for item in _EXPANSION_V3_FAMILIES
}:
    raise AssertionError("witness specs must exactly cover the v3 expansion families")
if len(semantic_family_definitions()) != 15:
    raise AssertionError("semantic-family universe v1 was unexpectedly mutated")
if set(BOUNDARY_PROFILE_IDS_BY_RISK_CLASS) != {
    item.risk_class for item in _PIPELINE_EVIDENCE
}:
    raise AssertionError("every issue risk class must declare boundary profiles")
