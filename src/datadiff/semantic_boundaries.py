from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datadiff.dsl import Case
from datadiff.dynamic_strategy import StrategyRuleRecord, load_strategy_snapshot
from datadiff.oracle import Finding
from datadiff.semantic_contracts import finding_contract_boundary_precedes_reference
from datadiff.util import unique_preserve_order

BoundaryPredicate = Callable[[Case, Finding | dict[str, Any], dict[str, Any]], bool]
PRE_REFERENCE_SEMANTIC_BOUNDARY_RULE_IDS = frozenset(
    {
        "boundary:join_null_keys",
        "boundary:modulo_semantics",
        "boundary:unicode_case_mapping",
        "boundary:special_float_filter_literal",
    }
)
SEMANTIC_CONTRACT_BOUNDARY_RULE_ID = "boundary:semantic_contract_lattice"


@dataclass(frozen=True, slots=True)
class SemanticBoundaryRule:
    rule_id: str
    reason: str
    predicate: BoundaryPredicate


@dataclass(frozen=True, slots=True)
class SemanticBoundaryMatch:
    rule_id: str
    reason: str


def pre_reference_semantic_boundary_matches(
    matches: list[SemanticBoundaryMatch],
    case: Case,
    finding: Finding | dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    """Select boundary evidence that must precede value-reference attribution."""

    kind = (
        finding.get("kind", "")
        if isinstance(finding, dict)
        else getattr(finding, "kind", "")
    )
    if str(kind or "") != "semantic_output_mismatch":
        return []
    return [
        match
        for match in matches
        if match.rule_id in PRE_REFERENCE_SEMANTIC_BOUNDARY_RULE_IDS
        or (
            match.rule_id == SEMANTIC_CONTRACT_BOUNDARY_RULE_ID
            and finding_contract_boundary_precedes_reference(case, finding)
        )
    ]


def finding_root_is(expected_root: str, get_finding_value: Callable[[Finding | dict[str, Any], str, Any], Any]) -> BoundaryPredicate:
    return lambda case, finding, config: str(get_finding_value(finding, "root_cause", "")) == expected_root


def build_documented_semantic_rules(
    *,
    documented_polars_nan_semantics: BoundaryPredicate,
) -> tuple[SemanticBoundaryRule, ...]:
    return (
        SemanticBoundaryRule(
            rule_id="documented:polars_nan_inf_semantics",
            reason="Polars documents NaN/NULL floating-point behavior for this mismatch class.",
            predicate=documented_polars_nan_semantics,
        ),
    )


def build_semantic_boundary_rules(
    *,
    get_finding_value: Callable[[Finding | dict[str, Any], str, Any], Any],
    case_contains_special_float: Callable[[Case], bool],
    join_null_semantics_boundary: BoundaryPredicate,
    null_filter_literal_boundary: BoundaryPredicate,
    special_float_filter_literal_boundary: BoundaryPredicate,
    modulo_boundary: BoundaryPredicate,
    unicode_case_mapping_boundary: BoundaryPredicate,
    semantic_contract_lattice_boundary: BoundaryPredicate | None = None,
) -> tuple[SemanticBoundaryRule, ...]:
    rules = [
        SemanticBoundaryRule(
            rule_id="boundary:root_nan_inf_semantics",
            reason="root cause nan_inf_semantics is a known cross-engine semantic boundary",
            predicate=finding_root_is("nan_inf_semantics", get_finding_value),
        ),
        SemanticBoundaryRule(
            rule_id="boundary:root_null_semantics",
            reason="root cause null_semantics is a known cross-engine semantic boundary",
            predicate=finding_root_is("null_semantics", get_finding_value),
        ),
        SemanticBoundaryRule(
            rule_id="boundary:root_ordering_or_limit",
            reason="root cause ordering_or_limit is a known cross-engine semantic boundary",
            predicate=finding_root_is("ordering_or_limit", get_finding_value),
        ),
        SemanticBoundaryRule(
            rule_id="boundary:special_float_values",
            reason="case contains NaN or Infinity values",
            predicate=lambda case, finding, config: (
                str(get_finding_value(finding, "root_cause", "")) == "nan_inf_semantics"
                and case_contains_special_float(case)
            ),
        ),
        SemanticBoundaryRule(
            rule_id="boundary:join_null_keys",
            reason="join key contains NULL values; NULL join semantics differ across target families",
            predicate=join_null_semantics_boundary,
        ),
        SemanticBoundaryRule(
            rule_id="boundary:null_filter_literal",
            reason="filter compares against NULL; engines intentionally differ on NULL predicate semantics",
            predicate=null_filter_literal_boundary,
        ),
        SemanticBoundaryRule(
            rule_id="boundary:special_float_filter_literal",
            reason=(
                "filter compares against NaN/Infinity; engines intentionally differ on "
                "special-float predicate semantics (NaN-as-value vs NaN-as-null)"
            ),
            predicate=special_float_filter_literal_boundary,
        ),
        SemanticBoundaryRule(
            rule_id="boundary:modulo_semantics",
            reason="case uses modulo; negative/float remainder semantics differ across engines",
            predicate=modulo_boundary,
        ),
        SemanticBoundaryRule(
            rule_id="boundary:unicode_case_mapping",
            reason="case changes case for non-ASCII text; Unicode case mapping support differs across engines",
            predicate=unicode_case_mapping_boundary,
        ),
    ]
    if semantic_contract_lattice_boundary is not None:
        rules.append(
            SemanticBoundaryRule(
                rule_id="boundary:semantic_contract_lattice",
                reason=(
                    "finding root/mismatch maps to a case semantic-contract axis "
                    "whose joined policy is boundary/probe"
                ),
                predicate=semantic_contract_lattice_boundary,
            )
        )
    return tuple(rules)


def semantic_rule_records(
    rules: tuple[SemanticBoundaryRule, ...],
    *,
    kind: str,
) -> tuple[StrategyRuleRecord, ...]:
    return tuple(
        StrategyRuleRecord(
            rule_id=rule.rule_id,
            kind=kind,
            reason=rule.reason,
            priority=100 + index,
        )
        for index, rule in enumerate(rules)
    )


def ordered_rules_from_snapshot(
    *,
    config: dict[str, Any],
    snapshot_key: str,
    default_rules: tuple[SemanticBoundaryRule, ...],
) -> tuple[SemanticBoundaryRule, ...]:
    snapshot = load_strategy_snapshot(str(config.get("strategy_snapshot_path", "") or ""))
    if snapshot is None:
        return default_rules
    requested_ids = [
        rule.rule_id
        for rule in (
            getattr(snapshot, snapshot_key, ())
            if hasattr(snapshot, snapshot_key)
            else ()
        )
        if isinstance(rule, StrategyRuleRecord) and rule.rule_id
    ]
    if not requested_ids:
        return default_rules
    by_id = {rule.rule_id: rule for rule in default_rules}
    ordered = [by_id[rule_id] for rule_id in requested_ids if rule_id in by_id]
    if not ordered:
        return default_rules
    ordered_ids = {rule.rule_id for rule in ordered}
    ordered.extend(rule for rule in default_rules if rule.rule_id not in ordered_ids)
    return tuple(ordered)


def matching_semantic_rules(
    rules: tuple[SemanticBoundaryRule, ...],
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    matches: list[SemanticBoundaryMatch] = []
    for rule in rules:
        if rule.predicate(case, finding, config):
            matches.append(SemanticBoundaryMatch(rule_id=rule.rule_id, reason=rule.reason))
    return matches


def semantic_boundary_reasons(matches: list[SemanticBoundaryMatch]) -> list[str]:
    return unique_preserve_order(match.reason for match in matches)


def documentation_refs(finding: Finding | dict[str, Any]) -> list[dict[str, str]]:
    root = str(
        finding.get("root_cause", "")
        if isinstance(finding, dict)
        else getattr(finding, "root_cause", "")
    )
    if root != "nan_inf_semantics":
        return []
    return [
        {
            "title": "Polars floating point numbers",
            "url": "https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#floating-point-numbers",
            "note": "Polars documents NaN ordering/comparison behavior as distinct from regular missing data.",
        },
        {
            "title": "Polars missing data",
            "url": "https://docs.pola.rs/user-guide/expressions/missing-data/#not-a-number-or-nan-values",
            "note": "Polars documents null as missing data and NaN as a floating-point value.",
        },
    ]
