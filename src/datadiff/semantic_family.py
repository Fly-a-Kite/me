from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from datadiff.util import unique_preserve_order


SEMANTIC_SIGNAL_PREFIXES = ("semantic_signal:", "combo_risk:")


@dataclass(frozen=True, slots=True)
class ScopedFragmentRule:
    prefixes: tuple[str, ...]
    fragments: tuple[str, ...]

    def matches(self, feature: str) -> bool:
        return feature.startswith(self.prefixes) and any(fragment in feature for fragment in self.fragments)


@dataclass(frozen=True, slots=True)
class SemanticFamilyRule:
    name: str
    exact_features: frozenset[str] = frozenset()
    prefix_features: tuple[str, ...] = ()
    scoped_fragments: tuple[ScopedFragmentRule, ...] = ()

    def matches(self, feature: str) -> bool:
        return (
            feature in self.exact_features
            or feature.startswith(self.prefix_features)
            or any(rule.matches(feature) for rule in self.scoped_fragments)
        )


SEMANTIC_FAMILY_RULES: tuple[SemanticFamilyRule, ...] = (
    SemanticFamilyRule(
        name="join_membership",
        exact_features=frozenset(
            {
                "pattern:semi_join_membership",
                "pattern:anti_join_exclusion",
                "pattern:semi_anti_join_rewrite",
                "pattern:semi_anti_join_null_keys",
            }
        ),
        prefix_features=("membership:",),
        scoped_fragments=(
            ScopedFragmentRule(
                (*SEMANTIC_SIGNAL_PREFIXES, "pattern:"),
                ("membership", "semi_anti_join", "semi_join", "anti_join", "join_key"),
            ),
            ScopedFragmentRule(("filter:",), ("set-membership", "negative-set-membership")),
        ),
    ),
    SemanticFamilyRule(
        name="null_semantics",
        exact_features=frozenset(
            {
                "pattern:fill_null_null_semantics",
                "pattern:coalesce_null_semantics",
                "pattern:drop_nulls_null_filter",
                "filter:null-predicate",
            }
        ),
        prefix_features=("null:", "nulls:", "fill_null:", "coalesce:"),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:", "filter:"), ("null", "coalesce", "fill_null")),
        ),
    ),
    SemanticFamilyRule(
        name="aggregation_cardinality",
        exact_features=frozenset(
            {
                "op:groupby",
                "op:aggregate",
                "pattern:string_count_groupby",
                "pattern:unique_count_groupby",
                "pattern:bool_null_groupby_agg",
            }
        ),
        prefix_features=("agg:", "groupby:"),
        scoped_fragments=(
            ScopedFragmentRule(
                (*SEMANTIC_SIGNAL_PREFIXES, "pattern:", "combo:"),
                ("aggregation", "groupby", "nunique", "distinct_count", "count_groupby"),
            ),
        ),
    ),
    SemanticFamilyRule(
        name="topk_ordering",
        exact_features=frozenset({"op:limit", "op:offset", "op:row_number_filter"}),
        prefix_features=("row_pick:", "sort:"),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:", "combo:"), ("topk", "ordering", "topn_per_group")),
        ),
    ),
    SemanticFamilyRule(
        name="string_semantics",
        exact_features=frozenset({"path:basename"}),
        prefix_features=("expr:string_", "string:", "path:"),
        scoped_fragments=(
            ScopedFragmentRule(
                (*SEMANTIC_SIGNAL_PREFIXES, "pattern:", "filter:", "common_api_template:"),
                ("string_", "string-", "basename", "normalized_string", "path_"),
            ),
        ),
    ),
    SemanticFamilyRule(
        name="type_coercion",
        exact_features=frozenset({"expr:cast", "float:literal-cast-consistency"}),
        prefix_features=("cast:", "cast_to:", "cast_domain:"),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:"), ("type_cast", "numeric_text_cast", "float_literal")),
        ),
    ),
    SemanticFamilyRule(
        name="boolean_logic",
        exact_features=frozenset({"expr:bool_not", "filter:truth-test", "filter:boolean-predicate"}),
        prefix_features=("nullable-bool:",),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:", "filter:"), ("boolean", "bool", "truth-test")),
        ),
    ),
    SemanticFamilyRule(
        name="conditional_semantics",
        exact_features=frozenset({"pattern:conditional_expression"}),
        prefix_features=("conditional:", "case_expr:"),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:"), ("case_when", "conditional")),
        ),
    ),
    SemanticFamilyRule(
        name="materialization_boundary",
        exact_features=frozenset(
            {
                "pattern:input_materialization_boundary",
                "pattern:filter_input_materialization",
                "pattern:cleanup_input_materialization",
            }
        ),
        prefix_features=("materialization:",),
        scoped_fragments=(
            ScopedFragmentRule(("pattern:",), ("input_materialization", "materialization")),
        ),
    ),
    SemanticFamilyRule(
        name="stateful_ordering",
        exact_features=frozenset({"op:running_sum", "op:row_number_filter", "op:sortedness_check"}),
        prefix_features=("running:", "row_pick:", "sortedness:", "window:", "quantile:"),
        scoped_fragments=(
            ScopedFragmentRule(
                (*SEMANTIC_SIGNAL_PREFIXES, "pattern:"),
                ("running_sum", "window_avg", "group_quantile", "sortedness", "keyed_pick", "topn_per_group"),
            ),
        ),
    ),
    SemanticFamilyRule(
        name="set_semantics",
        exact_features=frozenset({"op:distinct", "op:union_all", "pattern:distinct_deduplicate", "pattern:union_all_row_append"}),
        prefix_features=("distinct:", "union_all:", "setop:"),
        scoped_fragments=(
            ScopedFragmentRule((*SEMANTIC_SIGNAL_PREFIXES, "pattern:"), ("distinct", "union_all", "setop_all", "duplicate", "row_append")),
        ),
    ),
    SemanticFamilyRule(
        name="backend_specific_semantics",
        prefix_features=("pandas:", "polars:", "duckdb:", "pyarrow:", "datafusion:"),
        scoped_fragments=(
            ScopedFragmentRule(("pattern:",), ("pandas_", "polars_", "duckdb_", "pyarrow_", "datafusion_", "csv_")),
        ),
    ),
)


def derive_semantic_families(features: Iterable[str]) -> list[str]:
    normalized = [
        text
        for text in unique_preserve_order(str(feature).strip() for feature in features)
        if text
    ]
    if not normalized:
        return []
    matched: list[str] = []
    seen: set[str] = set()
    for feature in normalized:
        for family in _semantic_families_for_feature(feature):
            if family in seen:
                continue
            seen.add(family)
            matched.append(family)
    return matched


def derive_semantic_family_features(features: Iterable[str]) -> list[str]:
    return [f"semantic_family:{name}" for name in derive_semantic_families(features)]


@lru_cache(maxsize=None)
def _semantic_families_for_feature(feature: str) -> tuple[str, ...]:
    return tuple(
        rule.name
        for rule in SEMANTIC_FAMILY_RULES
        if rule.matches(feature)
    )
