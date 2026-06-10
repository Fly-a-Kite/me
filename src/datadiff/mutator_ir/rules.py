from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IR_REWRITE_RULE_REGISTRY_SCHEMA_VERSION = "ir-rewrite-rule-registry-v1"
SEMANTICS_PRESERVING = "semantics_preserving"
SEMANTICS_CHANGING_PROBE = "semantics_changing_probe"
OPTIMIZER_BOUNDARY = "optimizer_boundary"
IR_REWRITE_SEMANTICS_CLASSES = (
    SEMANTICS_PRESERVING,
    SEMANTICS_CHANGING_PROBE,
    OPTIMIZER_BOUNDARY,
)


@dataclass(frozen=True, slots=True)
class IRRewriteRule:
    rule_id: str
    operator: str
    relation: str
    semantics_class: str
    contract_axes: tuple[str, ...]
    optimizer_boundary: bool
    mutation_role: str
    metamorphic_role: str
    reducer_role: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "operator": self.operator,
            "relation": self.relation,
            "semantics_class": self.semantics_class,
            "contract_axes": list(self.contract_axes),
            "optimizer_boundary": self.optimizer_boundary,
            "mutation_role": self.mutation_role,
            "metamorphic_role": self.metamorphic_role,
            "reducer_role": self.reducer_role,
            "explanation": self.explanation,
        }


_IR_REWRITE_RULES: tuple[IRRewriteRule, ...] = (
    IRRewriteRule(
        rule_id="ir.rewrite.adjacent_independent_swap",
        operator="ir_swap_adjacent",
        relation="adjacent_independent_swap",
        semantics_class=SEMANTICS_PRESERVING,
        contract_axes=("ordering", "layout_sensitivity", "determinism"),
        optimizer_boundary=True,
        mutation_role="reorder independent local operations",
        metamorphic_role="same-result commutation relation",
        reducer_role="explain output-preserving local reorder",
        explanation=(
            "Swap adjacent row-local operations only when state signatures show the "
            "same available columns and types before/after the swap."
        ),
    ),
    IRRewriteRule(
        rule_id="ir.rewrite.filter_pushdown",
        operator="ir_pushdown_filter",
        relation="filter_pushdown",
        semantics_class=SEMANTICS_PRESERVING,
        contract_axes=("ordering", "null", "layout_sensitivity"),
        optimizer_boundary=True,
        mutation_role="move a valid filter before a dependency-safe producer",
        metamorphic_role="same-result optimizer pushdown relation",
        reducer_role="explain dependency-safe predicate movement",
        explanation=(
            "Move filters left across joins, projections, sorts, or distinct nodes "
            "only when read/produced columns remain valid in both orders."
        ),
    ),
    IRRewriteRule(
        rule_id="ir.rewrite.filter_above_groupby",
        operator="ir_pull_filter_above_groupby",
        relation="having_filter_source_probe",
        semantics_class=SEMANTICS_CHANGING_PROBE,
        contract_axes=("null", "dtype_coercion", "error_equivalence"),
        optimizer_boundary=True,
        mutation_role="probe HAVING-like aggregate/source predicate boundaries",
        metamorphic_role="not used as an equality oracle",
        reducer_role="explain intentional aggregate-boundary semantic shift",
        explanation=(
            "Rewrite a filter over an aggregate alias into a pre-groupby source-column "
            "predicate to probe aggregate pushdown boundary behavior."
        ),
    ),
    IRRewriteRule(
        rule_id="ir.rewrite.window_wrap",
        operator="ir_wrap_with_window",
        relation="window_boundary_probe",
        semantics_class=OPTIMIZER_BOUNDARY,
        contract_axes=("ordering", "null", "nan", "dtype_coercion"),
        optimizer_boundary=True,
        mutation_role="insert a valid ordered/window operation",
        metamorphic_role="not used as an equality oracle",
        reducer_role="explain added ordered/window boundary",
        explanation=(
            "Insert running-sum or row-number operations at legal program points to "
            "stress ordered partitions, null placement, and numeric window behavior."
        ),
    ),
    IRRewriteRule(
        rule_id="ir.rewrite.subtree_splice",
        operator="ir_splice_subtree",
        relation="local_subtree_splice",
        semantics_class=OPTIMIZER_BOUNDARY,
        contract_axes=("ordering", "layout_sensitivity", "determinism"),
        optimizer_boundary=True,
        mutation_role="move a locally valid operation subtree",
        metamorphic_role="optimizer-boundary variant, not an equality oracle by default",
        reducer_role="explain local subtree movement across legal positions",
        explanation=(
            "Move a small subtree of local relational operations to another valid "
            "program position to stress optimizer ordering boundaries."
        ),
    ),
    IRRewriteRule(
        rule_id="ir.rewrite.redundant_fold",
        operator="ir_fold_redundant_op",
        relation="redundant_operation_fold",
        semantics_class=SEMANTICS_PRESERVING,
        contract_axes=("ordering", "null", "layout_sensitivity"),
        optimizer_boundary=False,
        mutation_role="fold adjacent idempotent/redundant operations",
        metamorphic_role="same-result idempotence/fusion relation",
        reducer_role="explain shrink by redundant operation fold",
        explanation=(
            "Fold adjacent sort/select/limit/offset/drop-null operations where the "
            "merged operation preserves the relational result contract."
        ),
    ),
)
_RULE_BY_OPERATOR = {rule.operator: rule for rule in _IR_REWRITE_RULES}


def registered_ir_rewrite_rules() -> tuple[IRRewriteRule, ...]:
    return _IR_REWRITE_RULES


def ir_rewrite_rule_for_operator(operator: str) -> IRRewriteRule | None:
    return _RULE_BY_OPERATOR.get(str(operator or ""))


def ir_rewrite_rule_metadata(operator: str, *, detail: str = "") -> dict[str, Any]:
    rule = ir_rewrite_rule_for_operator(operator)
    if rule is None:
        return {}
    payload = rule.to_dict()
    if detail:
        payload["detail"] = str(detail)
    return payload


def ir_rewrite_rule_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": IR_REWRITE_RULE_REGISTRY_SCHEMA_VERSION,
        "semantics_classes": list(IR_REWRITE_SEMANTICS_CLASSES),
        "rules": [rule.to_dict() for rule in _IR_REWRITE_RULES],
        "methodology_claim": (
            "Typed IR rewrite rules are a single source for mutation, metamorphic "
            "relations, and reducer explanations; each rule declares whether it is "
            "semantics-preserving, a semantics-changing probe, or an optimizer boundary."
        ),
    }


def metamorphic_rewrite_rule_metadata() -> list[dict[str, Any]]:
    return [
        rule.to_dict()
        for rule in _IR_REWRITE_RULES
        if rule.semantics_class == SEMANTICS_PRESERVING
    ]
