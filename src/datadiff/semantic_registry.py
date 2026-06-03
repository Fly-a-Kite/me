from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    objective_feature,
    objective_features_for_rules,
)
from datadiff.semantic_family import SEMANTIC_FAMILY_RULES
from datadiff.targets import TargetContext
from datadiff.util import unique_preserve_order

SEMANTIC_REGISTRY_SCHEMA_VERSION = "semantic-registry-v1"
DEFAULT_ORACLE_ROLES: tuple[str, ...] = (
    "differential",
    "metamorphic",
    "semantic_boundary",
    "cross_version",
)
DEFAULT_EXTENSION_STEPS: tuple[str, ...] = (
    "declare_target_capabilities",
    "declare_semantic_objectives",
    "map_features_to_objectives",
    "map_mutation_operators_to_objectives",
    "attach_oracle_roles",
    "run_cross_backend_and_cross_version_matrix",
    "triage_reduce_recheck_and_record_ledger",
)


@dataclass(frozen=True, slots=True)
class SemanticObjectiveSpec:
    objective: str
    feature: str
    rules: tuple[ExplorationObjectiveRule, ...] = ()
    oracle_roles: tuple[str, ...] = DEFAULT_ORACLE_ROLES
    mutation_operator_affinity: tuple[str, ...] = ()
    capability_hints: tuple[str, ...] = ()
    methodology_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rules"] = [rule.to_dict() for rule in self.rules]
        return payload


@dataclass(frozen=True, slots=True)
class SemanticRegistry:
    objectives: tuple[SemanticObjectiveSpec, ...]
    semantic_families: tuple[str, ...]
    target_capabilities: tuple[str, ...] = ()
    generator_profiles: tuple[dict[str, Any], ...] = ()
    oracle_roles: tuple[str, ...] = DEFAULT_ORACLE_ROLES
    extension_steps: tuple[str, ...] = DEFAULT_EXTENSION_STEPS
    metadata: dict[str, Any] = field(default_factory=dict)

    def objective_features(self) -> list[str]:
        return [spec.feature for spec in self.objectives]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_REGISTRY_SCHEMA_VERSION,
            "objectives": [spec.to_dict() for spec in self.objectives],
            "semantic_families": list(self.semantic_families),
            "target_capabilities": list(self.target_capabilities),
            "generator_profiles": [dict(row) for row in self.generator_profiles],
            "oracle_roles": list(self.oracle_roles),
            "extension_steps": list(self.extension_steps),
            "metadata": dict(self.metadata),
            "methodology_claim": (
                "Targets are swapped through capability declarations and adapters; "
                "exploration is steered by neutral semantic objectives instead of "
                "backend-specific bug profiles."
            ),
        }


def build_semantic_registry(
    *,
    objective_rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None = None,
    target_context: TargetContext | Mapping[str, Any] | None = None,
    operator_profiles: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SemanticRegistry:
    rules = tuple(
        rule
        for rule in _coerce_objective_rules(objective_rules)
        if rule.objective
    )
    objective_features = objective_features_for_rules(rules)
    operator_affinity = _operator_affinity_by_objective(operator_profiles or {})
    specs = tuple(
        SemanticObjectiveSpec(
            objective=feature.removeprefix("exploration_objective:"),
            feature=feature,
            rules=tuple(rule for rule in rules if objective_feature(rule.objective) == feature),
            mutation_operator_affinity=tuple(operator_affinity.get(feature, ())),
            capability_hints=tuple(_capability_hints_for_objective(feature)),
            methodology_tags=tuple(_methodology_tags_for_objective(feature)),
        )
        for feature in objective_features
    )
    target_capabilities = _target_capabilities(target_context)
    return SemanticRegistry(
        objectives=specs,
        semantic_families=tuple(rule.name for rule in SEMANTIC_FAMILY_RULES),
        target_capabilities=tuple(target_capabilities),
        generator_profiles=tuple(_generator_profile_rows(target_capabilities)),
        metadata=dict(metadata or {}),
    )


def semantic_registry_payload(
    *,
    objective_rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None = None,
    target_context: TargetContext | Mapping[str, Any] | None = None,
    operator_profiles: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_semantic_registry(
        objective_rules=objective_rules,
        target_context=target_context,
        operator_profiles=operator_profiles,
        metadata=metadata,
    ).to_dict()


def _coerce_objective_rules(
    rules: Iterable[ExplorationObjectiveRule | Mapping[str, Any]] | None,
) -> list[ExplorationObjectiveRule]:
    out: list[ExplorationObjectiveRule] = []
    for rule in rules or ():
        if isinstance(rule, ExplorationObjectiveRule):
            out.append(rule)
        elif isinstance(rule, Mapping):
            out.append(ExplorationObjectiveRule(**dict(rule)))
    return out


def _operator_affinity_by_objective(operator_profiles: Mapping[str, Any]) -> dict[str, list[str]]:
    affinity: dict[str, list[str]] = {}
    for name, profile in operator_profiles.items():
        operator_name = str(name).strip()
        if not operator_name:
            continue
        for objective in getattr(profile, "exploration_objective_affinity", ()) or ():
            feature = objective_feature(objective)
            if not feature:
                continue
            affinity.setdefault(feature, []).append(operator_name)
    return {
        feature: unique_preserve_order(operators)
        for feature, operators in affinity.items()
    }


def _target_capabilities(target_context: TargetContext | Mapping[str, Any] | None) -> list[str]:
    if target_context is None:
        return []
    if isinstance(target_context, TargetContext):
        return list(target_context.common_capabilities)
    values = target_context.get("common_capabilities", target_context.get("shared_capabilities", []))
    if not isinstance(values, list | tuple):
        return []
    return unique_preserve_order([str(value).strip() for value in values if str(value).strip()])


def _capability_hints_for_objective(feature: str) -> list[str]:
    objective = feature.removeprefix("exploration_objective:")
    mapping = {
        "boundary_depth": ["op:limit", "op:offset", "op:sort", "op:mutate", "op:drop_nulls"],
        "cross_model_consistency": ["op:join", "op:groupby", "op:aggregate", "op:distinct"],
        "stateful_semantics": ["op:running_sum", "op:row_number_filter", "op:sortedness_check"],
        "representation_variance": ["op:mutate", "op:csv_long_numeric_roundtrip_probe", "op:struct_distinct_probe"],
        "predicate_logic": ["op:filter", "op:case_when", "op:tuple_absence_filter"],
        "coverage_breadth": ["table:single", "table:multi"],
    }
    return list(mapping.get(objective, []))


def _methodology_tags_for_objective(feature: str) -> list[str]:
    objective = feature.removeprefix("exploration_objective:")
    mapping = {
        "boundary_depth": ["semantic_boundary", "edge_condition"],
        "cross_model_consistency": ["cross_backend", "differential"],
        "stateful_semantics": ["stateful_relation", "metamorphic"],
        "representation_variance": ["adapter_boundary", "format_boundary"],
        "predicate_logic": ["logic_partition", "oracle_sensitivity"],
        "coverage_breadth": ["exploration", "portfolio"],
    }
    return list(mapping.get(objective, ["dynamic_objective"]))


def generator_profile_capability_requirements(profile: str) -> tuple[str, ...]:
    name = str(profile or "").strip()
    if not name:
        return ()
    tokens = set(name.split("_"))
    requirements: list[str] = ["table:single"]
    if "join" in tokens or "anti" in tokens or "membership" in tokens or "isin" in tokens:
        requirements.append("op:join")
    if "groupby" in tokens or "agg" in tokens or "aggregate" in tokens or "count" in tokens:
        requirements.extend(["op:groupby", "op:aggregate"])
    if "sort" in tokens or "sortedness" in tokens or "topk" in tokens or "order" in tokens or "ordered" in tokens:
        requirements.extend(["op:sort", "op:limit"])
    if "filter" in tokens or "predicate" in tokens or "truth" in tokens or "absence" in tokens:
        requirements.append("op:filter")
    if "cast" in tokens or "precision" in tokens or "round" in tokens or "float" in tokens:
        requirements.append("expr:cast")
    if "running" in tokens:
        requirements.append("op:running_sum")
    if "partitioned" in tokens or "partition" in tokens:
        requirements.append("running:partition_by")
    if "tuple" in tokens and "absence" in tokens:
        requirements.append("op:tuple_absence_filter")
    if "row" in tokens and "value" in tokens and "absence" in tokens:
        requirements.append("op:tuple_absence_filter")
    if "case" in tokens:
        requirements.append("op:case_when")
    if "coalesce" in tokens:
        requirements.append("op:coalesce")
    if "distinct" in tokens:
        requirements.append("op:distinct")
    if "union" in tokens or "setop" in tokens:
        requirements.append("op:union_all")
    if "json" in tokens:
        requirements.append("op:json_predicate_order_probe")
    if "sparse" in tokens:
        requirements.append("op:sparse_mask_probe")
    if "timestamp" in tokens:
        requirements.append("op:timestamp_precision_filter_probe")
    if "dataset" in tokens:
        requirements.append("op:dataset_isin_all_match_probe")
    if "run" in tokens and "end" in tokens:
        requirements.append("op:run_end_null_compute_probe")
    if "hash" in tokens and "pivot" in tokens:
        requirements.append("op:hash_pivot_wider_probe")
    if "list" in tokens and "flatten" in tokens:
        requirements.append("op:list_flatten_parent_indices_probe")
    if "rolling" in tokens:
        requirements.append("op:rolling_mean_by_null_count_probe")
    if "csv" in tokens:
        requirements.append("op:csv_long_numeric_roundtrip_probe")
    if "window" in tokens:
        requirements.append("op:window_avg_probe")
    if "struct" in tokens:
        requirements.append("op:struct_distinct_probe")
    if "bit" in tokens:
        requirements.append("op:bit_compare_probe")
    if "quantile" in tokens:
        requirements.append("op:group_quantile_probe")
    if "scalar" in tokens or "subquery" in tokens:
        requirements.append("op:scalar_subquery_probe")
    return tuple(unique_preserve_order(requirements))


def filter_generator_profiles_by_capability(
    profiles: Iterable[str],
    target_capabilities: Iterable[str],
) -> dict[str, Any]:
    capability_set = {str(capability).strip() for capability in target_capabilities if str(capability).strip()}
    normalized_profiles = unique_preserve_order([str(item).strip() for item in profiles if str(item).strip()])
    if not capability_set:
        return {
            "selected": normalized_profiles,
            "dropped": [],
            "target_capability_count": 0,
        }
    selected: list[str] = []
    dropped: list[dict[str, Any]] = []
    for profile in normalized_profiles:
        required = list(generator_profile_capability_requirements(profile))
        missing = [capability for capability in required if capability not in capability_set]
        if missing:
            dropped.append({"profile": profile, "required": required, "missing": missing})
            continue
        selected.append(profile)
    return {
        "selected": selected,
        "dropped": dropped,
        "target_capability_count": len(capability_set),
    }


def _generator_profile_rows(target_capabilities: list[str]) -> list[dict[str, Any]]:
    profile_names = (
        "common",
        "edge_float",
        "workflow",
        "discovery",
        "discovery_fresh",
        "discovery_no_groupby",
        "common_api_workflow",
        "issue_focus",
        "deep_probe_rotation",
    )
    capability_set = set(target_capabilities)
    rows: list[dict[str, Any]] = []
    for profile in profile_names:
        required = list(generator_profile_capability_requirements(profile))
        missing = [capability for capability in required if capability not in capability_set]
        rows.append(
            {
                "profile": profile,
                "required_capabilities": required,
                "supported_by_target": not missing if target_capabilities else True,
                "missing_capabilities": missing,
            }
        )
    return rows
