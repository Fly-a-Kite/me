from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from datadiff.util import unique_preserve_order

EXPLORATION_OBJECTIVE_PREFIX = "exploration_objective:"


def _normalize_objective_name(value: object) -> str:
    text = str(value).strip()
    if text.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        text = text.removeprefix(EXPLORATION_OBJECTIVE_PREFIX).strip()
    if text.startswith("objective:"):
        text = text.removeprefix("objective:").strip()
    return text.replace(" ", "_")


def _normalize_string_list(values: Iterable[Any] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


@dataclass(frozen=True, slots=True)
class ExplorationObjectiveRule:
    objective: str
    exact_features: frozenset[str] = frozenset()
    prefix_features: tuple[str, ...] = ()
    fragments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _normalize_objective_name(self.objective))
        object.__setattr__(
            self,
            "exact_features",
            frozenset(_normalize_string_list(self.exact_features)),
        )
        object.__setattr__(
            self,
            "prefix_features",
            tuple(_normalize_string_list(self.prefix_features)),
        )
        object.__setattr__(
            self,
            "fragments",
            tuple(_normalize_string_list(self.fragments)),
        )

    def matches(self, feature: str) -> bool:
        return (
            feature in self.exact_features
            or feature.startswith(self.prefix_features)
            or any(fragment in feature for fragment in self.fragments)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exact_features"] = sorted(self.exact_features)
        payload["prefix_features"] = list(self.prefix_features)
        payload["fragments"] = list(self.fragments)
        return payload

    def key(self) -> tuple[Any, ...]:
        return (
            self.objective,
            tuple(sorted(self.exact_features)),
            tuple(self.prefix_features),
            tuple(self.fragments),
        )

    def is_noop(self) -> bool:
        return not self.objective or (
            not self.exact_features
            and not self.prefix_features
            and not self.fragments
        )


DEFAULT_EXPLORATION_OBJECTIVE_RULES: tuple[ExplorationObjectiveRule, ...] = (
    ExplorationObjectiveRule(
        "boundary_depth",
        prefix_features=(
            "frontier:",
            "materialization:",
            "null:",
            "sort:",
            "row_pick:",
            "cast:",
        ),
        fragments=("boundary", "null", "precision", "topk", "offset", "limit", "cast"),
    ),
    ExplorationObjectiveRule(
        "cross_model_consistency",
        exact_features=frozenset(
            {
                "op:join",
                "op:groupby",
                "op:aggregate",
                "op:distinct",
                "op:union_all",
            }
        ),
        prefix_features=("semantic_family:join_membership", "semantic_family:aggregation_cardinality"),
        fragments=("join", "groupby", "aggregate", "distinct", "union_all", "membership"),
    ),
    ExplorationObjectiveRule(
        "stateful_semantics",
        exact_features=frozenset({"op:running_sum", "op:row_number_filter", "op:sortedness_check"}),
        prefix_features=("running:", "window:", "sortedness:", "row_pick:"),
        fragments=("running_sum", "window", "sortedness", "row_number", "quantile"),
    ),
    ExplorationObjectiveRule(
        "representation_variance",
        prefix_features=(
            "materialization:",
            "common_api_template:",
            "semantic_family:materialization_boundary",
            "semantic_family:type_coercion",
        ),
        fragments=("materialization", "cast", "string", "csv", "layout", "encoding", "dtype"),
    ),
    ExplorationObjectiveRule(
        "predicate_logic",
        exact_features=frozenset({"op:filter", "expr:bool_not"}),
        prefix_features=("filter:", "semantic_family:boolean_logic", "semantic_family:conditional_semantics"),
        fragments=("predicate", "truth", "boolean", "case_when", "conditional"),
    ),
    ExplorationObjectiveRule(
        "coverage_breadth",
        prefix_features=("opseq:", "combo:", "combo_frequency:", "op_count:"),
        fragments=("common_api", "workflow", "exploratory"),
    ),
)

EXPLORATION_OBJECTIVE_RULES = DEFAULT_EXPLORATION_OBJECTIVE_RULES
_RUNTIME_EXPLORATION_OBJECTIVE_RULES: list[ExplorationObjectiveRule] = []


def objective_feature(objective: str) -> str:
    text = _normalize_objective_name(objective)
    return f"{EXPLORATION_OBJECTIVE_PREFIX}{text}" if text else ""


def canonical_objective_key(value: object) -> str:
    text = str(value).strip()
    if text.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        return text
    if text.startswith("objective:"):
        objective = text.removeprefix("objective:").strip()
        return objective_feature(objective)
    return text


def coerce_exploration_objective_rule(
    value: ExplorationObjectiveRule | Mapping[str, Any],
) -> ExplorationObjectiveRule:
    if isinstance(value, ExplorationObjectiveRule):
        return ExplorationObjectiveRule(**value.to_dict())
    if isinstance(value, Mapping):
        return ExplorationObjectiveRule(**dict(value))
    raise TypeError(f"unsupported exploration objective rule payload: {type(value)!r}")


def merge_exploration_objective_rules(
    *groups: Iterable[ExplorationObjectiveRule | Mapping[str, Any] | None] | None,
) -> list[ExplorationObjectiveRule]:
    merged: list[ExplorationObjectiveRule] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        if group is None:
            continue
        for item in group:
            if item is None:
                continue
            rule = coerce_exploration_objective_rule(item)
            if rule.is_noop():
                continue
            key = rule.key()
            if key in seen:
                continue
            seen.add(key)
            merged.append(rule)
    return merged


def register_exploration_objective_rule(
    rule: ExplorationObjectiveRule | Mapping[str, Any],
) -> ExplorationObjectiveRule:
    normalized = coerce_exploration_objective_rule(rule)
    if normalized.is_noop():
        return normalized
    existing = {item.key() for item in _RUNTIME_EXPLORATION_OBJECTIVE_RULES}
    if normalized.key() not in existing:
        _RUNTIME_EXPLORATION_OBJECTIVE_RULES.append(normalized)
    return normalized


def clear_runtime_exploration_objective_rules() -> None:
    _RUNTIME_EXPLORATION_OBJECTIVE_RULES.clear()


def active_exploration_objective_rules(
    extra_rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None = None,
) -> list[ExplorationObjectiveRule]:
    return merge_exploration_objective_rules(
        DEFAULT_EXPLORATION_OBJECTIVE_RULES,
        _RUNTIME_EXPLORATION_OBJECTIVE_RULES,
        extra_rules,
    )


def derive_exploration_objectives(
    features: Iterable[str],
    *,
    rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None = None,
) -> list[str]:
    normalized = [
        text
        for text in unique_preserve_order(str(feature).strip() for feature in features)
        if text
    ]
    if not normalized:
        return []
    active_rules = active_exploration_objective_rules(rules)
    matched: list[str] = []
    seen: set[str] = set()
    for feature in normalized:
        for objective in _objectives_for_feature(feature, active_rules):
            if objective in seen:
                continue
            seen.add(objective)
            matched.append(objective)
    return matched


def derive_exploration_objective_features(
    features: Iterable[str],
    *,
    rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None = None,
) -> list[str]:
    return [
        objective_feature(objective)
        for objective in derive_exploration_objectives(features, rules=rules)
    ]


def objective_features_for_rules(
    rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None,
) -> list[str]:
    return unique_preserve_order([
        objective_feature(rule.objective)
        for rule in merge_exploration_objective_rules(rules)
        if rule.objective
    ])


def _objectives_for_feature(
    feature: str,
    rules: Iterable[ExplorationObjectiveRule],
) -> tuple[str, ...]:
    return tuple(
        rule.objective
        for rule in rules
        if rule.matches(feature)
    )

