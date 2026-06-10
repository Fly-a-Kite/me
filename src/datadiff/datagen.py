from __future__ import annotations

import random
import string
from typing import Any, Callable, Literal

from .common_api_workflow import COMMON_API_WORKFLOW_TEMPLATES, generate_common_api_workflow_case
from .discovery_profiles import (
    deep_probe_rotation_case as _deep_probe_rotation_profile_case,
    discovery_issue_inspired_case as _discovery_issue_inspired_profile_case,
    discovery_no_groupby_issue_inspired_case as _discovery_no_groupby_issue_inspired_profile_case,
    issue_focus_case as _issue_focus_profile_case,
)
from .dsl import Case, ColumnSpec, TableData
from .program_generation import generate_program, repair_operations
from .profile_generators import (
    generate_null_groupby_topk_case,
    generate_storage_offset_case,
    generate_null_agg_topk_case,
    generate_filter_null_agg_topk_case,
    generate_join_null_agg_topk_case,
    generate_join_null_key_topk_case,
    generate_wide_offset_topk_case,
    generate_empty_filter_groupby_case,
    generate_join_filter_groupby_case,
    generate_join_null_truth_filter_case,
    generate_join_groupby_stress_case,
    generate_float_group_key_case,
    generate_join_null_sort_case,
    generate_ordered_groupby_sort_case,
    generate_topk_resort_case,
    generate_join_ordered_agg_topk_case,
    generate_global_null_aggregate_case,
    generate_string_count_groupby_case,
    generate_unique_count_groupby_case,
    generate_bool_null_groupby_agg_case,
    generate_large_int_filter_groupby_case,
    generate_set_membership_filter_case,
    generate_pyarrow_groupby_filter_cast_membership_case,
    generate_null_predicate_filter_case,
    generate_boolean_predicate_filter_case,
    generate_post_topk_range_filter_case,
    generate_tuple_absence_filter_case,
    generate_row_value_absence_filter_case,
    generate_running_sum_precision_case,
    generate_partitioned_running_sum_case,
    generate_path_basename_keyed_pick_case,
    generate_sortedness_null_placement_case,
    generate_simple_case_random_subject_case,
    generate_group_quantile_key_probe_case,
    generate_scalar_subquery_double_parentheses_case,
    generate_window_avg_rows_frame_case,
    generate_struct_distinct_unnest_case,
    generate_bit_compare_unequal_length_case,
    generate_round_even_float_scale_case,
    generate_duckdb_float_literal_precision_case,
    generate_polars_timestamp_precision_filter_case,
    generate_series_rtruediv_operand_order_case,
    generate_polars_reverse_division_columns_case,
    generate_pandas_uint64_isin_precision_case,
    generate_duckdb_tuple_anti_null_semantics_case,
    generate_datafusion_setop_all_duplicate_count_case,
    generate_duckdb_json_predicate_order_semantics_case,
    generate_pandas_sparse_array_mask_semantics_case,
    generate_polars_float_wrap_numerical_semantics_case,
    generate_pandas_index_bool_result_type_case,
    generate_polars_empty_literal_groupby_semantics_case,
    generate_pandas_arrow_string_eq_sum_semantics_case,
    generate_pandas_arrow_timestamp_loc_slice_semantics_case,
    generate_pandas_arrow_timestamp_index_attr_semantics_case,
    generate_pandas_eval_inplace_aliasing_semantics_case,
    generate_pandas_bool_reduction_skipna_semantics_case,
    generate_pandas_arrow_bool_groupby_reduction_semantics_case,
    generate_pyarrow_dataset_isin_all_match_semantics_case,
    generate_pyarrow_run_end_null_compute_semantics_case,
    generate_pyarrow_large_string_partition_schema_semantics_case,
    generate_pyarrow_hash_pivot_wider_order_semantics_case,
    generate_pyarrow_list_flatten_parent_indices_semantics_case,
    generate_polars_rolling_mean_by_null_count_semantics_case,
    generate_csv_long_numeric_roundtrip_case,
)
from .synthesis.lhs_sampler import SchemaSpec
from .synthesis.typed_case import build_typed_grammar_case
from .workflow_profiles import generate_workflow_case

GeneratorProfile = Literal[
    "common",
    "edge_float",
    "typed_grammar",
    "workflow",
    "discovery",
    "discovery_fresh",
    "discovery_no_groupby",
    "common_api_workflow",
    "issue_focus",
    "deep_probe_rotation",
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "join_null_key_topk",
    "wide_offset_topk",
    "empty_filter_groupby",
    "join_filter_groupby",
    "join_null_truth_filter",
    "join_groupby_stress",
    "storage_offset",
    "float_group_key",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "unique_count_groupby",
    "bool_null_groupby_agg",
    "large_int_filter_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "boolean_predicate_filter",
    "post_topk_range_filter",
    "tuple_absence_filter",
    "row_value_absence_filter",
    "running_sum_precision",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "sortedness_null_placement",
    "simple_case_random_subject",
    "group_quantile_key_probe",
    "scalar_subquery_double_parentheses",
    "window_avg_rows_frame",
    "struct_distinct_unnest",
    "bit_compare_unequal_length",
    "round_even_float_scale",
    "duckdb_float_literal_precision",
    "polars_timestamp_precision_filter",
    "series_rtruediv_operand_order",
    "polars_reverse_division_columns",
    "pandas_uint64_isin_precision",
    "duckdb_tuple_anti_null_semantics",
    "datafusion_setop_all_duplicate_count",
    "duckdb_json_predicate_order_semantics",
    "pandas_sparse_array_mask_semantics",
    "polars_float_wrap_numerical_semantics",
    "pandas_index_bool_result_type",
    "polars_empty_literal_groupby_semantics",
    "pandas_arrow_string_eq_sum_semantics",
    "pandas_arrow_timestamp_loc_slice_semantics",
    "pandas_arrow_timestamp_index_attr_semantics",
    "pandas_eval_inplace_aliasing_semantics",
    "pandas_bool_reduction_skipna_semantics",
    "pandas_arrow_bool_groupby_reduction_semantics",
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_run_end_null_compute_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "pyarrow_list_flatten_parent_indices_semantics",
    "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip",
]


def _is_discovery_profile(profile: GeneratorProfile) -> bool:
    return profile in {"discovery", "discovery_fresh", "discovery_no_groupby", "issue_focus", "deep_probe_rotation"}


def _discovery_profile_allows_groupby(profile: GeneratorProfile) -> bool:
    return profile in {"discovery", "discovery_fresh", "issue_focus", "deep_probe_rotation"}


def _rand_str(rnd: random.Random) -> str | None:
    choices = ["", "alpha", "beta", "gamma", "δelta", "中文", "space value", "A", "a"]
    if rnd.random() < 0.7:
        return rnd.choice(choices)
    return "".join(rnd.choice(string.ascii_letters) for _ in range(rnd.randint(1, 8)))


def _value_for_type(
    rnd: random.Random,
    typ: str,
    nullable: bool = True,
    profile: GeneratorProfile = "common",
) -> Any:
    if nullable and rnd.random() < 0.18:
        return None
    if typ == "int":
        return rnd.choice([0, 1, -1, 2, -2, 10, -10, rnd.randint(-100, 100)])
    if typ == "float":
        if profile == "edge_float":
            special = rnd.random()
            if special < 0.05:
                return float("nan")
            if special < 0.08:
                return float("inf")
            if special < 0.11:
                return float("-inf")
        # The common profile targets a stable subset. NaN/Infinity are valuable,
        # but they belong in edge_float experiments because engines intentionally
        # disagree on their comparison semantics.
        return rnd.choice([0.0, 1.0, -1.0, 0.5, -0.5, round(rnd.uniform(-50, 50), 3)])
    if typ == "bool":
        return rnd.choice([True, False])
    if typ == "str":
        return _rand_str(rnd)
    raise ValueError(typ)


def generate_table(
    seed: int,
    name: str = "t0",
    min_rows: int = 0,
    max_rows: int = 20,
    profile: GeneratorProfile = "common",
) -> TableData:
    rnd = random.Random(seed)
    base_cols = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("y", "float", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
        ColumnSpec("s", "str", nullable=True),
    ]
    ncols = len(base_cols) if _is_discovery_profile(profile) else rnd.randint(3, len(base_cols))
    columns = base_cols[:ncols]
    nrows = rnd.randint(min_rows, max_rows)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        row: dict[str, Any] = {}
        for col in columns:
            if col.name == "id":
                # Repeated IDs are useful for joins and group-like behavior.
                choices = [i, i % 5, 0, 1]
                if _is_discovery_profile(profile):
                    choices.extend([i % 3, i % 2])
                row[col.name] = rnd.choice(choices)
            else:
                row[col.name] = _value_for_type(rnd, col.type, col.nullable, profile=profile)
        rows.append(row)
    return TableData(name=name, columns=columns, rows=rows)


def generate_table_from_schema_spec(
    seed: int,
    spec: SchemaSpec,
    *,
    name: str = "t0",
    profile: GeneratorProfile = "common",
) -> TableData:
    rnd = random.Random(seed * 1_000_003 + 97)
    column_count = max(2, int(spec.column_count))
    semantic_types = list(spec.type_mix) or ["numeric"]
    columns = [ColumnSpec("id", "int", nullable=False)]
    for index in range(1, column_count):
        semantic_type = semantic_types[index % len(semantic_types)]
        dsl_type = _schema_semantic_type_to_dsl_type(semantic_type, index)
        columns.append(ColumnSpec(_schema_column_name(dsl_type, index), dsl_type, nullable=True))
    rows: list[dict[str, Any]] = []
    for row_index in range(max(0, int(spec.row_count))):
        row: dict[str, Any] = {"id": rnd.choice([row_index, row_index % 5, 0, 1])}
        for column in columns[1:]:
            row[column.name] = _value_for_type(
                rnd,
                column.type,
                nullable=rnd.random() < max(0.0, min(1.0, float(spec.null_density))),
                profile=profile,
            )
        rows.append(row)
    return TableData(name=name, columns=columns, rows=rows)


def _schema_semantic_type_to_dsl_type(semantic_type: str, index: int) -> str:
    text = str(semantic_type).strip().lower()
    if text in {"string", "datetime"}:
        return "str"
    if text == "bool":
        return "bool"
    if text in {"float", "decimal"}:
        return "float"
    if text == "numeric":
        return "int" if index % 2 == 0 else "float"
    return "int"


def _schema_column_name(dsl_type: str, index: int) -> str:
    prefix = {"int": "i", "float": "f", "bool": "b", "str": "s"}.get(dsl_type, "c")
    if dsl_type == "str" and index % 3 == 0:
        prefix = "dt"
    return f"{prefix}{index}"


def generate_join_table(seed: int, profile: GeneratorProfile = "common") -> TableData:
    rnd = random.Random(seed * 3571 + 29)
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("j", "int", nullable=True),
        ColumnSpec("z", "float", nullable=True),
        ColumnSpec("tag", "str", nullable=True),
    ]
    nrows = rnd.randint(8, 18) if _is_discovery_profile(profile) else rnd.randint(1, 12)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        rows.append(
            {
                "id": (
                    rnd.choice([i, i % 5, i % 3, 0, 1, 2])
                    if _is_discovery_profile(profile)
                    else rnd.choice([i, i % 5, 0, 1, 2])
                ),
                "j": _value_for_type(rnd, "int", True, profile=profile),
                "z": _value_for_type(rnd, "float", True, profile=profile),
                "tag": _value_for_type(rnd, "str", True, profile=profile),
            }
        )
    return TableData(name="t1", columns=columns, rows=rows)


def _augment_join_table_with_primary_compat_columns(primary: TableData, right: TableData) -> TableData:
    existing = {column.name for column in right.columns}
    mirrored = [
        ColumnSpec(column.name, column.type, nullable=column.nullable)
        for column in primary.columns
        if column.name not in {"id", "row_nr"}
        and column.type in {"int", "str", "bool"}
        and column.name not in existing
    ]
    if not mirrored:
        return right
    primary_rows_by_id: dict[int, list[dict[str, Any]]] = {}
    for row in primary.rows:
        value = row.get("id")
        if isinstance(value, int) and not isinstance(value, bool):
            primary_rows_by_id.setdefault(value, []).append(row)
    rows: list[dict[str, Any]] = []
    fallback_rows = list(primary.rows)
    for idx, row in enumerate(right.rows):
        next_row = dict(row)
        source_rows = primary_rows_by_id.get(row.get("id"), fallback_rows)
        source_row = source_rows[idx % len(source_rows)] if source_rows else {}
        for column in mirrored:
            next_row[column.name] = source_row.get(column.name)
        rows.append(next_row)
    return TableData(right.name, [*right.columns, *mirrored], rows)




_TYPE_AWARE_PROFILE_GENERATORS: dict[str, Callable[[int], Case]] | None = None
_GENERIC_CASE_SUFFIXES = {
    "typed_grammar": "-typed-grammar",
    "discovery": "-discovery",
    "discovery_fresh": "-discovery-fresh",
    "discovery_no_groupby": "-discovery-no-groupby",
    "common_api_workflow": "-common-api-workflow",
    "issue_focus": "-issue-focus",
    "deep_probe_rotation": "-deep-probe-rotation",
}


def _type_aware_profile_generators() -> dict[str, Callable[[int], Case]]:
    global _TYPE_AWARE_PROFILE_GENERATORS
    if _TYPE_AWARE_PROFILE_GENERATORS is None:
        _TYPE_AWARE_PROFILE_GENERATORS = {
            "null_groupby_topk": generate_null_groupby_topk_case,
            "null_agg_topk": generate_null_agg_topk_case,
            "filter_null_agg_topk": generate_filter_null_agg_topk_case,
            "join_null_agg_topk": generate_join_null_agg_topk_case,
            "join_null_key_topk": generate_join_null_key_topk_case,
            "wide_offset_topk": generate_wide_offset_topk_case,
            "empty_filter_groupby": generate_empty_filter_groupby_case,
            "join_filter_groupby": generate_join_filter_groupby_case,
            "join_null_truth_filter": generate_join_null_truth_filter_case,
            "join_groupby_stress": generate_join_groupby_stress_case,
            "storage_offset": generate_storage_offset_case,
            "float_group_key": generate_float_group_key_case,
            "join_null_sort": generate_join_null_sort_case,
            "ordered_groupby_sort": generate_ordered_groupby_sort_case,
            "topk_resort": generate_topk_resort_case,
            "join_ordered_agg_topk": generate_join_ordered_agg_topk_case,
            "global_null_aggregate": generate_global_null_aggregate_case,
            "string_count_groupby": generate_string_count_groupby_case,
            "unique_count_groupby": generate_unique_count_groupby_case,
            "bool_null_groupby_agg": generate_bool_null_groupby_agg_case,
            "large_int_filter_groupby": generate_large_int_filter_groupby_case,
            "set_membership_filter": generate_set_membership_filter_case,
            "pyarrow_groupby_filter_cast_membership": generate_pyarrow_groupby_filter_cast_membership_case,
            "null_predicate_filter": generate_null_predicate_filter_case,
            "boolean_predicate_filter": generate_boolean_predicate_filter_case,
            "post_topk_range_filter": generate_post_topk_range_filter_case,
            "tuple_absence_filter": generate_tuple_absence_filter_case,
            "row_value_absence_filter": generate_row_value_absence_filter_case,
            "running_sum_precision": generate_running_sum_precision_case,
            "partitioned_running_sum": generate_partitioned_running_sum_case,
            "path_basename_keyed_pick": generate_path_basename_keyed_pick_case,
            "sortedness_null_placement": generate_sortedness_null_placement_case,
            "simple_case_random_subject": generate_simple_case_random_subject_case,
            "group_quantile_key_probe": generate_group_quantile_key_probe_case,
            "scalar_subquery_double_parentheses": generate_scalar_subquery_double_parentheses_case,
            "window_avg_rows_frame": generate_window_avg_rows_frame_case,
            "struct_distinct_unnest": generate_struct_distinct_unnest_case,
            "bit_compare_unequal_length": generate_bit_compare_unequal_length_case,
            "round_even_float_scale": generate_round_even_float_scale_case,
            "duckdb_float_literal_precision": generate_duckdb_float_literal_precision_case,
            "polars_timestamp_precision_filter": generate_polars_timestamp_precision_filter_case,
            "series_rtruediv_operand_order": generate_series_rtruediv_operand_order_case,
            "polars_reverse_division_columns": generate_polars_reverse_division_columns_case,
            "pandas_uint64_isin_precision": generate_pandas_uint64_isin_precision_case,
            "duckdb_tuple_anti_null_semantics": generate_duckdb_tuple_anti_null_semantics_case,
            "datafusion_setop_all_duplicate_count": generate_datafusion_setop_all_duplicate_count_case,
            "duckdb_json_predicate_order_semantics": generate_duckdb_json_predicate_order_semantics_case,
            "pandas_sparse_array_mask_semantics": generate_pandas_sparse_array_mask_semantics_case,
            "polars_float_wrap_numerical_semantics": generate_polars_float_wrap_numerical_semantics_case,
            "pandas_index_bool_result_type": generate_pandas_index_bool_result_type_case,
            "polars_empty_literal_groupby_semantics": generate_polars_empty_literal_groupby_semantics_case,
            "pandas_arrow_string_eq_sum_semantics": generate_pandas_arrow_string_eq_sum_semantics_case,
            "pandas_arrow_timestamp_loc_slice_semantics": generate_pandas_arrow_timestamp_loc_slice_semantics_case,
            "pandas_arrow_timestamp_index_attr_semantics": generate_pandas_arrow_timestamp_index_attr_semantics_case,
            "pandas_eval_inplace_aliasing_semantics": generate_pandas_eval_inplace_aliasing_semantics_case,
            "pandas_bool_reduction_skipna_semantics": generate_pandas_bool_reduction_skipna_semantics_case,
            "pandas_arrow_bool_groupby_reduction_semantics": (
                generate_pandas_arrow_bool_groupby_reduction_semantics_case
            ),
            "pyarrow_dataset_isin_all_match_semantics": generate_pyarrow_dataset_isin_all_match_semantics_case,
            "pyarrow_run_end_null_compute_semantics": generate_pyarrow_run_end_null_compute_semantics_case,
            "pyarrow_large_string_partition_schema_semantics": generate_pyarrow_large_string_partition_schema_semantics_case,
            "pyarrow_hash_pivot_wider_order_semantics": generate_pyarrow_hash_pivot_wider_order_semantics_case,
            "pyarrow_list_flatten_parent_indices_semantics": generate_pyarrow_list_flatten_parent_indices_semantics_case,
            "polars_rolling_mean_by_null_count_semantics": generate_polars_rolling_mean_by_null_count_semantics_case,
            "csv_long_numeric_roundtrip": generate_csv_long_numeric_roundtrip_case,
            "workflow": generate_workflow_case,
            "common_api_workflow": generate_common_api_workflow_case,
            "deep_probe_rotation": _deep_probe_rotation_case,
            "issue_focus": _issue_focus_case,
        }
    return _TYPE_AWARE_PROFILE_GENERATORS


def _profile_dispatch_case(seed: int, profile: str, *, type_aware: bool) -> Case | None:
    if not type_aware:
        return None
    if profile == "discovery":
        return _discovery_issue_inspired_case(seed)
    if profile == "discovery_no_groupby":
        mixed = _discovery_no_groupby_issue_inspired_case(seed)
        if mixed is not None:
            return mixed
    generator = _type_aware_profile_generators().get(profile)
    return generator(seed) if generator is not None else None


def _generic_case_suffix(profile: str) -> str:
    return _GENERIC_CASE_SUFFIXES.get(profile, "")


def generate_case(
    seed: int,
    type_aware: bool = True,
    profile: GeneratorProfile = "common",
    schema_spec: SchemaSpec | None = None,
) -> Case:
    dispatched = _profile_dispatch_case(seed, profile, type_aware=type_aware)
    if dispatched is not None:
        return dispatched
    discovery_profile = _is_discovery_profile(profile)
    typed_grammar_profile = profile == "typed_grammar"
    if schema_spec is not None:
        table = generate_table_from_schema_spec(seed, schema_spec, name="t0", profile=profile)
    else:
        table = generate_table(
            seed,
            name="t0",
            min_rows=8 if discovery_profile else 0,
            max_rows=30 if discovery_profile else 20,
            profile=profile,
        )
    rnd = random.Random(seed * 15485863 + 11)
    join_probability = 0.85 if discovery_profile else (0.65 if typed_grammar_profile else 0.4)
    extra_tables = [generate_join_table(seed, profile=profile)] if type_aware and rnd.random() < join_probability else []
    if discovery_profile and extra_tables:
        extra_tables = [
            _cover_join_table_keys(
                table,
                _augment_join_table_with_primary_compat_columns(table, extra_tables[0]),
            )
        ]
    if typed_grammar_profile and type_aware:
        return build_typed_grammar_case(
            seed,
            table,
            extra_tables=extra_tables,
            repair_operations_fn=repair_operations,
            schema_spec=schema_spec,
        )
    else:
        program = generate_program(
            seed,
            table,
            max_ops=8 if discovery_profile else 6,
            type_aware=type_aware,
            extra_tables=extra_tables,
            profile=profile,
        )
    suffix = _generic_case_suffix(profile)
    metadata = {}
    if schema_spec is not None:
        metadata["lhs_schema_spec"] = schema_spec.to_dict()
    return Case(
        case_id=f"case-{seed:08d}{suffix}",
        seed=seed,
        tables=[table] + extra_tables,
        program=program,
        metadata=metadata,
    )


def _deep_probe_rotation_case(seed: int) -> Case:
    return _deep_probe_rotation_profile_case(
        seed,
        generate_profile_case=lambda case_seed, profile: generate_case(case_seed, profile=profile),  # type: ignore[arg-type]
    )


def _issue_focus_case(seed: int) -> Case:
    return _issue_focus_profile_case(
        seed,
        generate_profile_case=lambda case_seed, profile: generate_case(case_seed, profile=profile),  # type: ignore[arg-type]
        profile_generators=_type_aware_profile_generators(),
    )


def _discovery_issue_inspired_case(seed: int) -> Case | None:
    return _discovery_issue_inspired_profile_case(
        seed,
        profile_generators=_type_aware_profile_generators(),
    )


def _discovery_no_groupby_issue_inspired_case(seed: int) -> Case | None:
    return _discovery_no_groupby_issue_inspired_profile_case(
        seed,
        profile_generators=_type_aware_profile_generators(),
    )


def _cover_join_table_keys(primary: TableData, right: TableData) -> TableData:
    existing = {row.get("id") for row in right.rows}
    rows = list(right.rows)
    primary_rows_by_id: dict[int, list[dict[str, Any]]] = {}
    for row in primary.rows:
        value = row.get("id")
        if isinstance(value, int) and not isinstance(value, bool):
            primary_rows_by_id.setdefault(value, []).append(row)
    left_ids = sorted(
        {
            row.get("id")
            for row in primary.rows
            if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
        }
    )
    for value in left_ids:
        if value in existing:
            continue
        source_rows = primary_rows_by_id.get(value, [])
        source_row = source_rows[0] if source_rows else {}
        row: dict[str, Any] = {}
        for column in right.columns:
            if column.name == "id":
                row[column.name] = value
            elif column.name in source_row:
                row[column.name] = source_row.get(column.name)
            elif column.type == "int":
                row[column.name] = value
            elif column.type == "float":
                row[column.name] = float(value)
            elif column.type == "bool":
                row[column.name] = bool(value % 2)
            else:
                row[column.name] = f"tag_{value}"
        rows.append(row)
        existing.add(value)
    return TableData(right.name, right.columns, rows)
