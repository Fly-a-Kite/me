from __future__ import annotations

import importlib
from functools import lru_cache
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datadiff.compact_novelty import DenseProductBitmap
from datadiff.dsl import Case
from datadiff.semantic_family_universe import (
    SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID,
)
from datadiff.semantic_family_universe_v2 import (
    SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
)
from datadiff.semantic_family_universe_v3 import (
    SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID,
    SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE,
    semantic_family_v3_definition,
    semantic_family_v3_witness_specs,
)


FAMILY_WITNESS_REGISTRY_SCHEMA_VERSION = "family-witness-registry-v1"
FAMILY_WITNESS_BITMAP_SCHEMA_VERSION = "family-witness-trigger-bitmap-v1"
GLOBAL_FAMILY_WITNESS_GENERATION_MODE = "goal_first_witness_global_v1"
GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID = "p8_semantic_witness_global_v1"
GLOBAL_FAMILY_WITNESS_TARGET_SUITE = "latest_all_engines"
GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE = SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE
GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID = SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID
GLOBAL_FAMILY_WITNESS_V2_TARGET_SUITE = "latest_all_engines"
GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE = SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID = SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID
GLOBAL_FAMILY_WITNESS_V3_TARGET_SUITE = "latest_all_engines"
GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE = SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE
GLOBAL_FAMILY_WITNESS_V4_METHOD_ARM_ID = SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID
GLOBAL_FAMILY_WITNESS_V4_TARGET_SUITE = SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE


@dataclass(frozen=True, slots=True)
class FamilyWitnessRegistration:
    family_id: str
    generation_mode: str
    method_arm_id: str
    goal_id: str
    root_id: str
    target_suite: str
    backends: tuple[str, ...]
    target_backend: str
    axes: tuple[tuple[str, tuple[str, ...]], ...]
    builder: str
    backend_scope_policy: str = "full"
    screening_plan_policy: str = "finding_only"
    cache_reuse_source_family_id: str = ""
    cache_reuse_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family_id or not self.generation_mode or not self.goal_id:
            raise ValueError("family witness identity fields must be non-empty")
        if not self.axes or any(not name or not values for name, values in self.axes):
            raise ValueError("family witness axes must be non-empty")
        axis_names = [name for name, _values in self.axes]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("family witness axis names must be unique")
        if any(len(set(values)) != len(values) for _name, values in self.axes):
            raise ValueError("family witness axis values must be unique")
        if self.target_backend not in self.backends:
            raise ValueError(
                "family witness target backend must belong to registered backends"
            )
        if self.backend_scope_policy not in {"full", "balanced_axis_coverage"}:
            raise ValueError(
                "unsupported family witness backend scope policy: "
                f"{self.backend_scope_policy}"
            )
        if self.screening_plan_policy != "finding_only":
            raise ValueError(
                "registered family witnesses must defer full plan collection "
                "until a finding"
            )
        if bool(self.cache_reuse_source_family_id) != bool(self.cache_reuse_axes):
            raise ValueError(
                "cache reuse source family and axes must be declared together"
            )
        if self.cache_reuse_source_family_id == self.family_id:
            raise ValueError("family witness cannot reuse itself as a cache source")
        if not set(self.cache_reuse_axes) <= set(axis_names):
            raise ValueError("cache reuse axes must belong to the registered axes")

    @property
    def cell_count(self) -> int:
        count = 1
        for _name, values in self.axes:
            count *= len(values)
        return count

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(name for name, _values in self.axes)

    @property
    def axis_values(self) -> tuple[tuple[str, ...], ...]:
        return tuple(values for _name, values in self.axes)

    def cell_for_seed(self, seed: int) -> tuple[int, dict[str, str]]:
        cell_index = int(seed) % self.cell_count
        remainder = cell_index
        coordinates: dict[str, str] = {}
        strides: list[int] = []
        stride = 1
        for _name, values in reversed(self.axes):
            strides.append(stride)
            stride *= len(values)
        strides.reverse()
        for (name, values), axis_stride in zip(self.axes, strides, strict=True):
            value_index, remainder = divmod(remainder, axis_stride)
            coordinates[name] = values[value_index]
        return cell_index, coordinates

    def cell_index_for_axes(self, axes: Mapping[str, Any]) -> int:
        normalized = {str(name): str(value) for name, value in axes.items()}
        if set(normalized) != set(self.axis_names):
            raise ValueError(
                f"family witness axes mismatch for {self.family_id}: "
                f"expected={self.axis_names}, observed={tuple(normalized)}"
            )
        index = 0
        for name, values in self.axes:
            try:
                value_index = values.index(normalized[name])
            except ValueError as exc:
                raise ValueError(
                    f"unknown axis value for {self.family_id}: "
                    f"{name}={normalized[name]}"
                ) from exc
            index = index * len(values) + value_index
        return index

    def generate_case(self, seed: int, *, profile: str = "") -> Case:
        module_name, separator, function_name = self.builder.rpartition(".")
        if not separator:
            raise ValueError(f"invalid family witness builder: {self.builder}")
        module = importlib.import_module(module_name)
        builder = getattr(module, function_name)
        generated = builder(int(seed), profile=str(profile or ""))
        return generated.case if hasattr(generated, "case") else generated

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": FAMILY_WITNESS_REGISTRY_SCHEMA_VERSION,
            "family_id": self.family_id,
            "generation_mode": self.generation_mode,
            "method_arm_id": self.method_arm_id,
            "goal_id": self.goal_id,
            "root_id": self.root_id,
            "target_suite": self.target_suite,
            "backends": list(self.backends),
            "target_backend": self.target_backend,
            "axes": [
                {"name": name, "values": list(values)}
                for name, values in self.axes
            ],
            "cell_count": self.cell_count,
            "builder": self.builder,
            "backend_scope_policy": self.backend_scope_policy,
            "screening_plan_policy": self.screening_plan_policy,
            "cache_reuse_source_family_id": self.cache_reuse_source_family_id,
            "cache_reuse_axes": list(self.cache_reuse_axes),
        }


@dataclass(frozen=True, slots=True)
class FamilyWitnessObservation:
    registered: bool
    family_id: str
    cell_index: int | None
    first_seen: bool | None
    axes: dict[str, str]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FAMILY_WITNESS_BITMAP_SCHEMA_VERSION,
            "registered": self.registered,
            "family_id": self.family_id,
            "cell_index": self.cell_index,
            "first_seen": self.first_seen,
            "axes": dict(self.axes),
            "reason": self.reason,
        }


class FamilyWitnessBitmap:
    __slots__ = ("registration", "_product", "unregistered_observations")

    def __init__(self, registration: FamilyWitnessRegistration) -> None:
        self.registration = registration
        self._product = DenseProductBitmap(registration.axis_values)
        self.unregistered_observations = 0

    def observe_case(self, case: Case) -> FamilyWitnessObservation:
        metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
        witness = metadata.get("family_witness", {})
        if not isinstance(witness, Mapping):
            witness = {}
        family_id = str(witness.get("family_id", "") or "")
        raw_axes = witness.get("axes", {})
        axes = (
            {str(key): str(value) for key, value in raw_axes.items()}
            if isinstance(raw_axes, Mapping)
            else {}
        )
        coordinates = tuple(axes.get(name, "") for name in self.registration.axis_names)
        if family_id != self.registration.family_id:
            return self._unregistered(
                family_id,
                axes,
                f"family_id_mismatch:{family_id}",
            )
        try:
            cell_index, first_seen = self._product.observe(coordinates)
        except (KeyError, ValueError) as exc:
            return self._unregistered(
                family_id,
                axes,
                f"unregistered_axis_value:{exc}",
            )
        declared_index = witness.get("cell_index")
        if declared_index is not None and int(declared_index) != cell_index:
            return self._unregistered(
                family_id,
                axes,
                "cell_index_mismatch:"
                f"declared={declared_index},encoded={cell_index}",
                cell_index=cell_index,
                first_seen=first_seen,
            )
        return FamilyWitnessObservation(
            registered=True,
            family_id=family_id,
            cell_index=cell_index,
            first_seen=first_seen,
            axes=axes,
        )

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        return {
            **self._product.snapshot(include_data=include_data),
            "schema_version": FAMILY_WITNESS_BITMAP_SCHEMA_VERSION,
            "family_id": self.registration.family_id,
            "generation_mode": self.registration.generation_mode,
            "dimension_names": list(self.registration.axis_names),
            "dimension_values": [
                list(values) for values in self.registration.axis_values
            ],
            "unregistered_observation_count": self.unregistered_observations,
            "behavioral_role": "shadow_first_seen_hint_only",
            "case_discard_authority": False,
            "confirmed_bug_dedupe_authority": False,
        }

    def _unregistered(
        self,
        family_id: str,
        axes: dict[str, str],
        reason: str,
        *,
        cell_index: int | None = None,
        first_seen: bool | None = None,
    ) -> FamilyWitnessObservation:
        self.unregistered_observations += 1
        return FamilyWitnessObservation(
            registered=False,
            family_id=family_id,
            cell_index=cell_index,
            first_seen=first_seen,
            axes=axes,
            reason=reason,
        )


_CONFIRMED_ROOT_REGISTRATIONS: tuple[FamilyWitnessRegistration, ...] = (
    FamilyWitnessRegistration(
        family_id="pyarrow_sliced_bool_groupby_any_all",
        generation_mode="goal_first_witness_v4",
        method_arm_id="p8_semantic_witness_v4",
        goal_id="pyarrow_layout_bool_groupby",
        root_id="pyarrow-sliced-bool-hash-aggregate-001",
        target_suite="pandas_pyarrow",
        backends=("pandas", "pyarrow"),
        target_backend="pyarrow",
        axes=(
            ("layout", ("contiguous", "sliced", "chunked")),
            ("variant", ("any_all", "all_any")),
            (
                "data_pattern",
                (
                    "single_group_false_null",
                    "two_groups_false_null",
                    "null_key_false_null",
                ),
            ),
        ),
        builder=(
            "datadiff.semantic_layout_witness."
            "generate_physical_layout_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_reflected_arithmetic_operand_order",
        generation_mode="goal_first_witness_v5",
        method_arm_id="p8_semantic_witness_v5",
        goal_id="polars_reflected_arithmetic",
        root_id="polars-reflected-arithmetic-operand-order-001",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars",
        axes=(
            (
                "operator",
                ("rsub", "rtruediv", "rfloordiv", "rmod", "rpow"),
            ),
            ("name_mode", ("distinct_names", "shared_name")),
            (
                "data_pattern",
                ("canonical_positive", "unit_boundary", "mixed_order"),
            ),
        ),
        builder=(
            "datadiff.semantic_reflected_arithmetic_witness."
            "generate_reflected_arithmetic_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_grouped_null_topk",
        generation_mode="goal_first_witness_v6",
        method_arm_id="p8_semantic_witness_v6",
        goal_id="datafusion_grouped_null_topk",
        root_id="datafusion-grouped-null-topk-001",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("aggregate", ("min", "max")),
            (
                "exposure_mode",
                ("full_nulls_last", "top1_nulls_first"),
            ),
            (
                "data_pattern",
                (
                    "singleton_null",
                    "null_value_groups",
                    "repeated_null_group",
                ),
            ),
        ),
        builder=(
            "datadiff.semantic_datafusion_grouped_null_topk_witness."
            "generate_datafusion_grouped_null_topk_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_limit_offset_pushdown",
        generation_mode=GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        goal_id="datafusion_limit_offset_pushdown",
        root_id="datafusion-limit-offset-pushdown-001",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("inner_limit", ("limit8", "limit16")),
            ("outer_offset", ("offset1", "offset2")),
            (
                "data_pattern",
                (
                    "four_groups",
                    "duplicate_join_match",
                    "nullable_join_payload",
                ),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_datafusion_limit_offset_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_negative_zero_comparison",
        generation_mode=GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        goal_id="datafusion_negative_zero_comparison",
        root_id="datafusion-negative-zero-comparison-001",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("expression_mode", ("multiply_neg_one", "unary_negate")),
            ("comparator", ("ge_zero", "eq_zero")),
            (
                "data_pattern",
                (
                    "singleton_zero",
                    "duplicate_zero",
                    "zero_with_positive_control",
                ),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_datafusion_negative_zero_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_distinct_null_topk",
        generation_mode=GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        goal_id="datafusion_distinct_null_topk",
        root_id="datafusion-distinct-null-topk-001",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("order_mode", ("asc_nulls_first", "desc_nulls_first")),
            ("projection_mode", ("direct", "aliased_subquery")),
            (
                "data_pattern",
                ("null_empty_text", "duplicate_null", "mixed_text"),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_datafusion_distinct_null_topk_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_ordered_limit_idempotence",
        generation_mode=GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        goal_id="datafusion_ordered_limit_idempotence",
        root_id="datafusion-ordered-limit-idempotence-001",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("limit_n", ("limit5", "limit6")),
            ("offset_n", ("offset1", "offset2")),
            (
                "data_pattern",
                ("canonical_rows", "reversed_input", "duplicate_boundary"),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_datafusion_limit_idempotence_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_grouped_max_sort_metadata",
        generation_mode=GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        goal_id="polars_grouped_max_sort_metadata",
        root_id="polars-grouped-max-sort-metadata-001",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars",
        axes=(
            (
                "input_order",
                ("asc_nulls_last", "asc_nulls_first"),
            ),
            ("group_shape", ("two_groups", "three_groups")),
            (
                "data_pattern",
                ("canonical_counts", "duplicate_max", "null_heavy"),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_polars_grouped_max_metadata_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="duckdb_join_filter_pushdown_limit",
        generation_mode="goal_first_witness_v7",
        method_arm_id="p8_semantic_witness_v7",
        goal_id="duckdb_join_filter_pushdown_limit",
        root_id="duckdb-join-filter-pushdown-limit-001",
        target_suite="embedded_sql_cross",
        backends=("pandas", "duckdb", "sqlite"),
        target_backend="duckdb",
        axes=(
            ("flag_mode", ("canonical", "mirrored")),
            ("cut_mode", ("offset2_asc", "offset1_desc")),
            ("membership_mode", ("inner_join", "semi_join")),
            (
                "data_pattern",
                (
                    "canonical_counts",
                    "duplicate_low_groups",
                    "duplicate_selected_group",
                ),
            ),
        ),
        builder=(
            "datadiff.semantic_confirmed_root_witness."
            "generate_duckdb_join_filter_witness_case"
        ),
    ),
)

_EXPANSION_REGISTRATIONS: tuple[FamilyWitnessRegistration, ...] = (
    FamilyWitnessRegistration(
        family_id="pandas_nullable_bool_reduction",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="pandas_nullable_bool_reduction",
        root_id="target-family-pandas-nullable-bool-reduction-v1",
        target_suite="dataframe",
        backends=("pandas", "polars"),
        target_backend="pandas",
        axes=(
            ("context", ("frame", "groupby")),
            ("reduction", ("any", "all")),
            ("data_pattern", ("true_null", "false_null", "all_null")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_pandas_nullable_bool_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="sqlite_affinity_null_ordered_cut",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="sqlite_affinity_null_ordered_cut",
        root_id="target-family-sqlite-affinity-null-ordered-cut-v1",
        target_suite="embedded_sql_cross",
        backends=("pandas", "duckdb", "sqlite"),
        target_backend="sqlite",
        axes=(
            ("cast_mode", ("numeric_text_to_int", "int_to_text")),
            ("null_order", ("first", "last")),
            ("data_pattern", ("boundary", "duplicates", "nullable")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_sqlite_affinity_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_lazy_filter_groupby_window",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="polars_lazy_filter_groupby_window",
        root_id="target-family-polars-lazy-filter-groupby-window-v1",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars_lazy",
        axes=(
            ("filter_mode", ("ge_zero", "le_two")),
            ("window_order", ("asc", "desc")),
            ("data_pattern", ("balanced", "duplicate_order", "nullable")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_polars_lazy_optimizer_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_lazy_temporal_cast_boundary",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="polars_lazy_temporal_cast_boundary",
        root_id="target-family-polars-lazy-temporal-cast-boundary-v1",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars_lazy",
        axes=(
            ("temporal_op", ("precision_cast", "timezone_convert")),
            ("direction", ("forward", "reverse")),
            ("boundary", ("below", "equal", "above")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_polars_lazy_temporal_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="pyarrow_encoded_nested_compute",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="pyarrow_encoded_nested_compute",
        root_id="target-family-pyarrow-encoded-nested-compute-v1",
        target_suite="pandas_pyarrow",
        backends=("pandas", "pyarrow"),
        target_backend="pyarrow",
        axes=(
            (
                "probe",
                ("run_end_null", "list_parent", "large_string_partition"),
            ),
            ("layout", ("chunked", "dictionary")),
            ("data_pattern", ("null_heavy", "duplicate_boundary")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_pyarrow_encoded_nested_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_window_order_join_interaction",
        generation_mode=GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
        goal_id="datafusion_window_order_join_interaction",
        root_id="target-family-datafusion-window-order-join-interaction-v1",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("join_mode", ("inner", "left")),
            ("order_mode", ("asc", "desc")),
            ("data_pattern", ("balanced", "duplicate_join", "nullable_payload")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_witness."
            "generate_datafusion_window_join_witness_case"
        ),
    ),
)

_EXPANSION_V2_REGISTRATIONS: tuple[FamilyWitnessRegistration, ...] = (
    FamilyWitnessRegistration(
        family_id="pandas_nullable_string_normalization",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="pandas_nullable_string_normalization",
        root_id="target-family-pandas-nullable-string-normalization-v2",
        target_suite="pandas_pyarrow",
        backends=("pandas", "pyarrow"),
        target_backend="pandas",
        axes=(
            ("transform", ("lower_strip", "upper_replace")),
            ("null_policy", ("coalesce", "fill_null")),
            (
                "data_pattern",
                ("unicode_empty", "null_duplicates", "whitespace_case"),
            ),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_pandas_nullable_string_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="pyarrow_string_layout_predicate",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="pyarrow_string_layout_predicate",
        root_id="target-family-pyarrow-string-layout-predicate-v2",
        target_suite="pandas_pyarrow",
        backends=("pandas", "pyarrow"),
        target_backend="pyarrow",
        axes=(
            ("layout", ("contiguous", "chunked", "dictionary")),
            ("predicate", ("contains", "starts_with", "ends_with")),
            ("data_pattern", ("ascii_nullable", "unicode_duplicates")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_pyarrow_string_layout_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_arithmetic_cast_sortedness",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="polars_arithmetic_cast_sortedness",
        root_id="target-family-polars-arithmetic-cast-sortedness-v2",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars",
        axes=(
            ("arithmetic", ("add_clip", "abs_string_cast")),
            ("order_mode", ("asc_nulls_last", "desc_nulls_first")),
            ("data_pattern", ("boundary", "duplicates", "nullable")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_polars_arithmetic_sortedness_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="polars_lazy_case_string_membership",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="polars_lazy_case_string_membership",
        root_id="target-family-polars-lazy-case-string-membership-v2",
        target_suite="dataframe_lazy",
        backends=("polars", "polars_lazy"),
        target_backend="polars_lazy",
        axes=(
            ("membership", ("semi_join", "anti_join")),
            ("string_expr", ("lower_concat", "slice_replace")),
            ("data_pattern", ("duplicate_keys", "null_keys", "unicode")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_polars_lazy_string_membership_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="duckdb_union_duplicate_global_aggregate",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="duckdb_union_duplicate_global_aggregate",
        root_id="target-family-duckdb-union-duplicate-global-aggregate-v2",
        target_suite="embedded_sql_cross",
        backends=("pandas", "duckdb", "sqlite"),
        target_backend="duckdb",
        axes=(
            ("aggregate_pair", ("count_sum", "mean_nunique")),
            ("dedupe", ("raw_union", "distinct_union")),
            ("data_pattern", ("duplicates", "nulls", "boundary")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_duckdb_union_aggregate_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="sqlite_three_valued_membership",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="sqlite_three_valued_membership",
        root_id="target-family-sqlite-three-valued-membership-v2",
        target_suite="embedded_sql_cross",
        backends=("pandas", "duckdb", "sqlite"),
        target_backend="sqlite",
        axes=(
            (
                "membership",
                ("semi_join", "anti_join", "tuple_absence_filter"),
            ),
            ("key_shape", ("single", "tuple")),
            ("data_pattern", ("right_null", "left_null")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_sqlite_three_valued_membership_witness_case"
        ),
    ),
    FamilyWitnessRegistration(
        family_id="datafusion_null_setop_aggregate",
        generation_mode=GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
        goal_id="datafusion_null_setop_aggregate",
        root_id="target-family-datafusion-null-setop-aggregate-v2",
        target_suite="datafusion_cross",
        backends=("pandas", "duckdb", "datafusion"),
        target_backend="datafusion",
        axes=(
            ("null_op", ("drop_nulls", "fill_null", "coalesce")),
            ("aggregate_pair", ("sum_min", "mean_count")),
            ("data_pattern", ("all_null_group", "duplicate_null")),
        ),
        builder=(
            "datadiff.semantic_family_expansion_v2_witness."
            "generate_datafusion_null_setop_witness_case"
        ),
    ),
)

_EXPANSION_V3_BUILDERS = {
    "cross_backend_issue_risk_pipeline": (
        "datadiff.semantic_family_expansion_v3_witness."
        "generate_cross_backend_issue_risk_witness_case"
    ),
    "cross_backend_exact_dtype_pipeline": (
        "datadiff.semantic_family_expansion_v3_witness."
        "generate_cross_backend_exact_dtype_witness_case"
    ),
    "pyarrow_layout_interaction_pipeline": (
        "datadiff.semantic_family_expansion_v3_witness."
        "generate_pyarrow_layout_interaction_witness_case"
    ),
}

_EXPANSION_V3_REGISTRATIONS: tuple[FamilyWitnessRegistration, ...] = tuple(
    FamilyWitnessRegistration(
        family_id=spec.family_id,
        generation_mode=GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE,
        method_arm_id=GLOBAL_FAMILY_WITNESS_V4_METHOD_ARM_ID,
        goal_id=spec.family_id,
        root_id=semantic_family_v3_definition(spec.family_id).root_id,
        target_suite=GLOBAL_FAMILY_WITNESS_V4_TARGET_SUITE,
        backends=spec.execution_backends,
        target_backend=semantic_family_v3_definition(spec.family_id).target_backend,
        axes=spec.axes,
        builder=_EXPANSION_V3_BUILDERS[spec.family_id],
        backend_scope_policy=spec.backend_scope_policy,
        screening_plan_policy=spec.screening_plan_policy,
        cache_reuse_source_family_id=spec.cache_reuse_source_family_id,
        cache_reuse_axes=spec.cache_reuse_axes,
    )
    for spec in semantic_family_v3_witness_specs()
)

_V1_ALL_REGISTRATIONS = (
    *_CONFIRMED_ROOT_REGISTRATIONS,
    *_EXPANSION_REGISTRATIONS,
)

_V2_ALL_REGISTRATIONS = (
    *_V1_ALL_REGISTRATIONS,
    *_EXPANSION_V2_REGISTRATIONS,
)

_ALL_REGISTRATIONS = (
    *_V2_ALL_REGISTRATIONS,
    *_EXPANSION_V3_REGISTRATIONS,
)

_GLOBAL_PORTFOLIOS: dict[str, tuple[FamilyWitnessRegistration, ...]] = {
    GLOBAL_FAMILY_WITNESS_GENERATION_MODE: _CONFIRMED_ROOT_REGISTRATIONS,
    GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE: _V1_ALL_REGISTRATIONS,
    GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE: _V2_ALL_REGISTRATIONS,
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE: _ALL_REGISTRATIONS,
}

_GLOBAL_METHOD_ARMS = {
    GLOBAL_FAMILY_WITNESS_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
    GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V2_METHOD_ARM_ID,
    GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V3_METHOD_ARM_ID,
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V4_METHOD_ARM_ID,
}

_GLOBAL_TARGET_SUITES = {
    GLOBAL_FAMILY_WITNESS_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_TARGET_SUITE,
    GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V2_TARGET_SUITE,
    GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V3_TARGET_SUITE,
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE: GLOBAL_FAMILY_WITNESS_V4_TARGET_SUITE,
}

_BY_MODE: dict[str, tuple[FamilyWitnessRegistration, ...]] = {}
for _registration in _ALL_REGISTRATIONS:
    _BY_MODE[_registration.generation_mode] = (
        *_BY_MODE.get(_registration.generation_mode, ()),
        _registration,
    )
_BY_FAMILY = {
    registration.family_id: registration for registration in _ALL_REGISTRATIONS
}

for _registration in _ALL_REGISTRATIONS:
    if not _registration.cache_reuse_source_family_id:
        continue
    try:
        _source_registration = _BY_FAMILY[
            _registration.cache_reuse_source_family_id
        ]
    except KeyError as exc:
        raise ValueError(
            "unknown family witness cache reuse source: "
            f"{_registration.cache_reuse_source_family_id}"
        ) from exc
    _source_axes = dict(_source_registration.axes)
    _consumer_axes = dict(_registration.axes)
    if set(_registration.cache_reuse_axes) != set(_source_registration.axis_names):
        raise ValueError(
            f"cache reuse axes must fully identify source cells: "
            f"{_registration.family_id}"
        )
    for _axis_name in _registration.cache_reuse_axes:
        if _axis_name not in _source_axes:
            raise ValueError(
                f"cache reuse axis {_axis_name} is missing from source family "
                f"{_source_registration.family_id}"
            )
        if not set(_consumer_axes[_axis_name]) <= set(_source_axes[_axis_name]):
            raise ValueError(
                f"cache reuse axis values exceed source for {_registration.family_id}:"
                f"{_axis_name}"
            )
    if not set(_registration.backends) <= set(_source_registration.backends):
        raise ValueError(
            f"cache consumer backends exceed source scope: {_registration.family_id}"
        )


def family_witness_registrations() -> tuple[FamilyWitnessRegistration, ...]:
    """Return the frozen global-v1 confirmed-root registrations."""

    return _CONFIRMED_ROOT_REGISTRATIONS


def all_family_witness_registrations() -> tuple[FamilyWitnessRegistration, ...]:
    """Return the frozen v1-universe registrations used by global-v2."""

    return _V1_ALL_REGISTRATIONS


def latest_family_witness_registrations() -> tuple[FamilyWitnessRegistration, ...]:
    """Return every registration through semantic-family universe v3."""

    return _ALL_REGISTRATIONS


def global_family_witness_registrations(
    generation_mode: str = GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
) -> tuple[FamilyWitnessRegistration, ...]:
    try:
        return _GLOBAL_PORTFOLIOS[str(generation_mode)]
    except KeyError as exc:
        raise KeyError(f"unknown global family witness mode: {generation_mode}") from exc


def family_witness_registration_for_mode(
    generation_mode: str,
) -> FamilyWitnessRegistration | None:
    registrations = _BY_MODE.get(str(generation_mode or ""), ())
    return registrations[0] if len(registrations) == 1 else None


def family_witness_registrations_for_mode(
    generation_mode: str,
) -> tuple[FamilyWitnessRegistration, ...]:
    return _BY_MODE.get(str(generation_mode or ""), ())


def family_witness_registration(
    family_id: str,
) -> FamilyWitnessRegistration:
    return _BY_FAMILY[str(family_id)]


def family_witness_registration_for_case(
    case: Case,
) -> FamilyWitnessRegistration | None:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping):
        return None
    return _BY_FAMILY.get(str(witness.get("family_id", "") or ""))


def family_witness_execution_backends(case: Case) -> tuple[str, ...]:
    """Return the declarative low-I/O screening scope for a family witness.

    The registration retains the complete backend universe used by exhaustive
    audits. A family may additionally choose a deterministic coverage-preserving
    schedule for repeated screening; the schedule never invents backends and
    keeps a multi-backend comparison on every cell.
    """

    registration = family_witness_registration_for_case(case)
    if registration is None:
        return ()
    if registration.backend_scope_policy == "balanced_axis_coverage":
        metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
        witness = metadata.get("family_witness", {})
        if isinstance(witness, Mapping):
            raw_index = witness.get("cell_index")
            try:
                cell_index = int(raw_index)
            except (TypeError, ValueError):
                cell_index = -1
            schedule = _balanced_axis_backend_schedule(registration.family_id)
            if 0 <= cell_index < len(schedule):
                return schedule[cell_index]
    return registration.backends


def family_witness_screening_plan_policy(case: Case) -> str:
    """Return the registered plan-evidence policy for screening execution."""

    registration = family_witness_registration_for_case(case)
    return "method" if registration is None else registration.screening_plan_policy


@lru_cache(maxsize=None)
def _balanced_axis_backend_schedule(
    family_id: str,
) -> tuple[tuple[str, ...], ...]:
    registration = family_witness_registration(family_id)
    if registration.backend_scope_policy != "balanced_axis_coverage":
        return tuple(registration.backends for _ in range(registration.cell_count))

    participants = registration.backends
    minimum_backends_per_cell = min(3, len(participants))

    cells: list[dict[str, Any]] = []
    primary_axis = "pipeline" if "pipeline" in registration.axis_names else registration.axis_names[0]
    for cell_index in range(registration.cell_count):
        case = registration.generate_case(cell_index)
        _resolved_index, axes = registration.cell_for_seed(cell_index)
        boundary = (
            case.metadata.get("boundary_application", {})
            if isinstance(case.metadata, Mapping)
            else {}
        )
        boundary_profile = (
            str(boundary.get("profile_id", "") or "")
            if isinstance(boundary, Mapping) and boundary.get("applied") is True
            else ""
        )
        cells.append(
            {
                "cell_index": cell_index,
                "primary_axis": str(axes.get(primary_axis, "")),
                "boundary_profile": boundary_profile,
            }
        )

    assignments: list[set[str]] = [set() for _ in cells]
    primary_values = list(dict(registration.axes)[primary_axis])
    primary_groups: dict[str, list[int]] = {}
    for primary_index, primary_value in enumerate(primary_values):
        group = [
            int(cell["cell_index"])
            for cell in cells
            if cell["primary_axis"] == primary_value
        ]
        primary_groups[str(primary_value)] = group
        if not group:
            continue
        for backend_index, backend in enumerate(participants):
            slot = (backend_index + primary_index) % len(group)
            assignments[group[slot]].add(backend)

    boundary_profiles = sorted(
        {
            str(cell["boundary_profile"])
            for cell in cells
            if cell["boundary_profile"]
        }
    )
    for boundary_profile in boundary_profiles:
        profile_cells = [
            int(cell["cell_index"])
            for cell in cells
            if cell["boundary_profile"] == boundary_profile
        ]
        for backend in participants:
            if any(backend in assignments[index] for index in profile_cells):
                continue
            movable: list[tuple[int, int]] = []
            for selected_index in profile_cells:
                primary_value = str(cells[selected_index]["primary_axis"])
                current_index = next(
                    (
                        index
                        for index in primary_groups[primary_value]
                        if backend in assignments[index]
                    ),
                    -1,
                )
                if (
                    current_index >= 0
                    and current_index != selected_index
                    and len(assignments[current_index])
                    > minimum_backends_per_cell
                ):
                    movable.append((selected_index, current_index))
            if movable:
                selected_index, current_index = min(
                    movable,
                    key=lambda item: (
                        len(assignments[item[0]]),
                        -len(assignments[item[1]]),
                        item[0],
                    ),
                )
                assignments[current_index].remove(backend)
                assignments[selected_index].add(backend)
            else:
                selected_index = min(
                    profile_cells,
                    key=lambda index: (len(assignments[index]), index),
                )
                assignments[selected_index].add(backend)

    backend_order = {backend: index for index, backend in enumerate(registration.backends)}
    return tuple(
        tuple(
            sorted(
                selected,
                key=backend_order.__getitem__,
            )
        )
        for selected in assignments
    )


def generate_registered_family_witness_case(
    generation_mode: str,
    seed: int,
    *,
    profile: str = "",
) -> Case | None:
    if str(generation_mode or "") in _GLOBAL_PORTFOLIOS:
        return generate_global_family_witness_case(
            seed,
            profile=profile,
            generation_mode=str(generation_mode),
        )
    registration = family_witness_registration_for_mode(generation_mode)
    if registration is None:
        return None
    return registration.generate_case(seed, profile=profile)


@dataclass(frozen=True, slots=True)
class GlobalFamilyWitnessSelection:
    generation_mode: str
    method_arm_id: str
    target_suite: str
    global_cell_index: int
    global_cell_count: int
    registration: FamilyWitnessRegistration
    family_cell_index: int
    family_construction_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "global-family-witness-selection-v1",
            "generation_mode": self.generation_mode,
            "method_arm_id": self.method_arm_id,
            "target_suite": self.target_suite,
            "global_cell_index": self.global_cell_index,
            "global_cell_count": self.global_cell_count,
            "family_id": self.registration.family_id,
            "root_id": self.registration.root_id,
            "family_cell_index": self.family_cell_index,
            "family_cell_count": self.registration.cell_count,
            "family_construction_seed": self.family_construction_seed,
        }


def global_family_witness_cell_count(
    generation_mode: str = GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
) -> int:
    return sum(
        registration.cell_count
        for registration in global_family_witness_registrations(generation_mode)
    )


def global_family_witness_selection(
    seed: int,
    *,
    generation_mode: str = GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
) -> GlobalFamilyWitnessSelection:
    registrations = global_family_witness_registrations(generation_mode)
    total = global_family_witness_cell_count(generation_mode)
    if total <= 0:
        raise RuntimeError("global family witness registry is empty")
    global_index = int(seed) % total
    remainder = global_index
    for registration in registrations:
        if remainder < registration.cell_count:
            cycle = int(seed) // total
            construction_seed = cycle * registration.cell_count + remainder
            return GlobalFamilyWitnessSelection(
                generation_mode=generation_mode,
                method_arm_id=_GLOBAL_METHOD_ARMS[generation_mode],
                target_suite=_GLOBAL_TARGET_SUITES[generation_mode],
                global_cell_index=global_index,
                global_cell_count=total,
                registration=registration,
                family_cell_index=remainder,
                family_construction_seed=construction_seed,
            )
        remainder -= registration.cell_count
    raise AssertionError("global family witness cell selection overflow")


def generate_global_family_witness_case(
    seed: int,
    *,
    profile: str = "",
    generation_mode: str = GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
) -> Case:
    selection = global_family_witness_selection(
        seed,
        generation_mode=generation_mode,
    )
    case = selection.registration.generate_case(
        selection.family_construction_seed,
        profile=profile,
    )
    source_generation_mode = str(
        case.metadata.get("generation_mode", "")
        if isinstance(case.metadata, Mapping)
        else ""
    )
    case.seed = int(seed)
    case.program.seed = int(seed)
    case.case_id = (
        f"case-{int(seed):08d}-global-{selection.registration.family_id}-"
        f"{selection.family_cell_index:03d}"
    )
    case.program.program_id = (
        f"prog-{int(seed):08d}-global-{selection.registration.family_id}-"
        f"{selection.family_cell_index:03d}"
    )
    case.metadata = dict(case.metadata or {})
    case.metadata["generation_mode"] = generation_mode
    global_payload = {
        **selection.to_dict(),
        "source_generation_mode": source_generation_mode,
        "requested_seed": int(seed),
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    case.metadata["global_family_witness"] = global_payload
    family_witness = case.metadata.get("family_witness", {})
    if isinstance(family_witness, dict):
        family_witness["source_generation_mode"] = str(
            family_witness.get("generation_mode", source_generation_mode) or ""
        )
        family_witness["generation_mode"] = generation_mode
        family_witness["global_cell_index"] = selection.global_cell_index
        family_witness["global_cell_count"] = selection.global_cell_count
    trace = case.metadata.get("goal_first_generation", {})
    if isinstance(trace, dict):
        trace["source_generation_mode"] = str(
            trace.get("generation_mode", source_generation_mode) or ""
        )
        trace["generation_mode"] = generation_mode
        trace["global_family_witness"] = global_payload
    return case
