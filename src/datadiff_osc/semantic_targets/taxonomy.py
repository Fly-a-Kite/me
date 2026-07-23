from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.rules import (
    AGGREGATE_KINDS,
    EXPRESSION_KINDS,
    OPERATION_KINDS,
)


ATOM_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]*:[^\s:][^\s]*$")
ATOM_NAMESPACES = frozenset(
    {
        "op", "expr", "agg", "chain", "type", "cast", "data", "order",
        "rows", "partition", "layout", "mode", "boundary", "oracle", "axis",
        "pipeline", "risk", "capability", "contract", "source",
    }
)

KNOWN_PROBE_OPERATIONS = frozenset(
    {
        "arrow_bool_groupby_reduction_probe", "arrow_string_contains_na_probe",
        "arrow_string_eq_sum_probe", "arrow_timestamp_index_attr_probe",
        "arrow_timestamp_loc_slice_probe", "bit_compare_probe",
        "bool_reduction_skipna_probe",
        "csv_long_numeric_roundtrip_probe", "datafusion_grouped_null_topk_probe",
        "dataset_isin_all_match_probe", "empty_literal_groupby_probe",
        "eval_inplace_alias_probe", "float_literal_precision_probe",
        "float_wrap_probe", "group_quantile_probe", "hash_pivot_wider_probe",
        "index_bool_probe", "json_predicate_order_probe",
        "large_string_partition_probe", "list_flatten_parent_indices_probe",
        "polars_timezone_filter_probe", "random_case_probe",
        "rolling_mean_by_null_count_probe", "round_even_probe",
        "run_end_null_compute_probe", "scalar_subquery_probe",
        "series_reflected_arithmetic_probe", "series_rtruediv_probe",
        "setop_all_duplicate_probe", "sparse_mask_probe", "struct_distinct_probe",
        "timestamp_precision_filter_probe", "tuple_anti_null_probe",
        "uint64_isin_probe", "window_avg_probe",
    }
)


# Axis values lower to observable semantics.  The table is deliberately keyed by
# generic axis/value pairs and never by a family, backend, root or candidate ID.
_VALUE_ATOMS: dict[str, frozenset[str]] = {
    "count_sum": frozenset({"op:aggregate", "agg:count", "agg:sum"}),
    "mean_nunique": frozenset({"op:aggregate", "agg:mean", "agg:nunique"}),
    "sum_min": frozenset({"agg:sum", "agg:min"}),
    "mean_count": frozenset({"agg:mean", "agg:count"}),
    "add_clip": frozenset({"op:mutate", "expr:add_const", "expr:clip"}),
    "abs_string_cast": frozenset({"op:mutate", "expr:abs", "expr:cast"}),
    "lower_strip": frozenset({"op:mutate", "expr:string_lower", "expr:string_strip"}),
    "upper_replace": frozenset({"op:mutate", "expr:string_upper", "expr:string_replace"}),
    "lower_concat": frozenset({"op:mutate", "expr:string_lower", "expr:string_concat"}),
    "slice_replace": frozenset({"op:mutate", "expr:string_slice", "expr:string_replace"}),
    "numeric_text_to_int": frozenset({"op:mutate", "expr:cast", "cast:int"}),
    "int_to_text": frozenset({"op:mutate", "expr:cast", "cast:str"}),
    "raw_union": frozenset({"op:union_all"}),
    "distinct_union": frozenset({"op:union_all", "op:distinct"}),
    "coalesce": frozenset({"op:coalesce"}),
    "fill_null": frozenset({"op:fill_null"}),
    "drop_nulls": frozenset({"op:drop_nulls"}),
    "semi_join": frozenset({"op:semi_join"}),
    "anti_join": frozenset({"op:anti_join"}),
    "tuple_absence_filter": frozenset({"op:tuple_absence_filter"}),
    "contains": frozenset({"expr:string_contains"}),
    "starts_with": frozenset({"expr:string_starts_with"}),
    "ends_with": frozenset({"expr:string_ends_with"}),
    "precision_cast": frozenset({"op:timestamp_precision_filter_probe", "expr:cast"}),
    "timezone_convert": frozenset({"op:polars_timezone_filter_probe"}),
    "true_null": frozenset({"data:bool_true_with_null"}),
    "false_null": frozenset({"data:bool_false_with_null"}),
    "all_null": frozenset({"data:all_null_column"}),
    "boundary": frozenset({"data:numeric_boundary"}),
    "duplicates": frozenset({"data:duplicate_values"}),
    "nullable": frozenset({"data:null_values"}),
    "balanced": frozenset({"data:balanced_groups"}),
    "duplicate_order": frozenset({"data:duplicate_order_keys"}),
    "null_heavy": frozenset({"data:null_heavy"}),
    "duplicate_boundary": frozenset({"data:duplicate_boundary"}),
    "duplicate_join": frozenset({"data:duplicate_join_keys"}),
    "nullable_payload": frozenset({"data:nullable_payload"}),
    "unicode_empty": frozenset({"data:unicode", "data:empty_string"}),
    "null_duplicates": frozenset({"data:null_values", "data:duplicate_strings"}),
    "whitespace_case": frozenset({"data:whitespace", "data:case_variants"}),
    "ascii_nullable": frozenset({"data:ascii", "data:null_values"}),
    "unicode_duplicates": frozenset({"data:unicode", "data:duplicate_strings"}),
    "duplicate_keys": frozenset({"data:duplicate_normalized_keys"}),
    "null_keys": frozenset({"data:null_membership_keys"}),
    "unicode": frozenset({"data:unicode"}),
    "nulls": frozenset({"data:null_values"}),
    "right_null": frozenset({"data:right_key_null"}),
    "left_null": frozenset({"data:left_key_null"}),
    "all_null_group": frozenset({"data:all_null_group"}),
    "duplicate_null": frozenset({"data:duplicate_null_row"}),
    "baseline": frozenset({"data:baseline_palette"}),
    "reordered_duplicate": frozenset({"data:stress_palette"}),
}

_AXIS_NAMESPACES: dict[str, str] = {
    "layout": "layout",
    "physical_layout": "layout",
    "execution_mode": "mode",
    "mode": "mode",
    "boundary": "boundary",
    "direction": "boundary",
    "null_order": "order",
    "order_mode": "order",
    "window_order": "order",
    "input_order": "order",
    "data_pattern": "data",
    "risk": "risk",
}


class UnknownAtom(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AtomSet:
    tokens: frozenset[str]
    integer_ids: frozenset[int]
    taxonomy_digest: str

    def __contains__(self, token: str) -> bool:
        return token in self.tokens

    @property
    def digest(self) -> str:
        return stable_digest("osc-atom-set", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class AtomTaxonomy:
    namespaces: frozenset[str] = ATOM_NAMESPACES
    operation_kinds: frozenset[str] = frozenset((*OPERATION_KINDS, *KNOWN_PROBE_OPERATIONS))
    expression_kinds: frozenset[str] = frozenset(EXPRESSION_KINDS)
    aggregate_kinds: frozenset[str] = frozenset(AGGREGATE_KINDS)
    schema_version: str = "osc-atom-taxonomy-v1"

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-atom-taxonomy",
            {
                "schema_version": self.schema_version,
                "namespaces": sorted(self.namespaces),
                "operations": sorted(self.operation_kinds),
                "expressions": sorted(self.expression_kinds),
                "aggregates": sorted(self.aggregate_kinds),
            },
        )

    def validate(self, token: str) -> str:
        text = str(token).strip()
        if not ATOM_TOKEN_RE.fullmatch(text):
            raise UnknownAtom(f"malformed semantic atom: {token!r}")
        namespace, value = text.split(":", 1)
        if namespace not in self.namespaces:
            raise UnknownAtom(f"unknown semantic atom namespace: {namespace}")
        if namespace == "op" and value not in self.operation_kinds:
            raise UnknownAtom(f"unknown operation atom: {value}")
        if namespace == "expr" and value not in self.expression_kinds:
            raise UnknownAtom(f"unknown expression atom: {value}")
        if namespace == "agg" and value not in self.aggregate_kinds:
            raise UnknownAtom(f"unknown aggregate atom: {value}")
        return text

    def atom_id(self, token: str) -> int:
        text = self.validate(token)
        digest = hashlib.sha256(
            f"{self.schema_version}\0{text}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def compile(self, tokens: Iterable[str]) -> AtomSet:
        validated = frozenset(self.validate(item) for item in tokens)
        ids = [self.atom_id(item) for item in validated]
        if len(ids) != len(set(ids)):
            raise RuntimeError("semantic atom integer-ID collision")
        return AtomSet(validated, frozenset(ids), self.digest)


def axis_token(name: str, value: str) -> str:
    return f"axis:{name}={value}"


def semantic_atoms_for_axis(name: str, value: str) -> frozenset[str]:
    """Generic axis-to-atom lowering; it never reads a family identifier."""

    atoms = set(_VALUE_ATOMS.get(value, ()))
    # Data-pattern coordinates are proven by row-derived facts rather than by a
    # selected coordinate label.  Other axes are structural program/layout facts.
    if name != "data_pattern":
        atoms.add(axis_token(name, value))
    if value in OPERATION_KINDS:
        atoms.add(f"op:{value}")
    if value in EXPRESSION_KINDS:
        atoms.add(f"expr:{value}")
    if value in AGGREGATE_KINDS:
        atoms.add(f"agg:{value}")
    if name in {"layout", "physical_layout"}:
        atoms.add(f"layout:{value}")
    if name in {"execution_mode", "mode"}:
        atoms.add(f"mode:{value}")
    if name == "pipeline":
        atoms.add(f"pipeline:{value}")
    if name == "boundary":
        atoms.add(f"data:{value}")
    namespace = _AXIS_NAMESPACES.get(name)
    if namespace is not None:
        if name != "data_pattern":
            atoms.add(f"{namespace}:{value}")
    if name in {"membership", "null_op"} and value in OPERATION_KINDS:
        atoms.add(f"op:{value}")
    if name == "reduction" and value in AGGREGATE_KINDS:
        atoms.add(f"agg:{value}")
    if name == "probe":
        probe_map = {
            "run_end_null": "run_end_null_compute_probe",
            "list_parent": "list_flatten_parent_indices_probe",
            "large_string_partition": "large_string_partition_probe",
        }
        if value in probe_map:
            atoms.add(f"op:{probe_map[value]}")
    return frozenset(atoms)
