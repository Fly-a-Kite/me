from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
import math
import random
from itertools import combinations
from dataclasses import dataclass, field
from datadiff.champion_corpus import ChampionSeed, ChampionRegistry
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from datadiff.datagen import repair_operations
from datadiff.disagreement import DisagreementDescriptor
from datadiff.dsl import Case, ColumnSpec, IRNode, Program, TableData, normalize_sort_keys
from datadiff.energy import operator_energy
from datadiff.identifiers import make_safe_output_name
from datadiff.join_keys import join_key_arg, join_key_pairs
from datadiff.mutator_ir import (
    apply_adjacent_independent_swap,
    apply_filter_pushdown,
    apply_filter_above_groupby,
    apply_redundant_op_fold,
    apply_subtree_splice,
    apply_wrap_with_window,
    ir_rewrite_rule_metadata,
)
from datadiff.operation_semantics import aggregate_alias, aggregate_column, aggregate_func, aggregate_specs, op_ascending, op_column, op_kind, op_n, operation_count
from datadiff.program_state import ProgramState, state_after_operations
from datadiff.mutator_shrink import (
    shrink_drop_tail_op,
    shrink_fold_redundant_op,
    shrink_inline_single_use_mutate,
    shrink_merge_adjacent_filters,
)
from datadiff.util import unique_preserve_order
from datadiff.value_catalog import CatalogEntry, sample as sample_catalog_entry

INHERITED_METADATA_KEYS = (
    "generator_profile",
    "mixed_generator_profile",
    "source_issue",
    "source_issue_alt",
)

BOOLEAN_PROBE_OUTPUT_PREFIXES = (
    "sorted_ok_",
    "unexpected_else_seen",
    "quantile_key_mismatch",
    "scalar_subquery_mismatch",
    "window_avg_mismatch",
    "struct_distinct_mismatch",
    "bit_compare_mismatch",
    "round_even_mismatch",
    "series_rtruediv_mismatch",
    "uint64_isin_mismatch",
    "tuple_anti_null_mismatch",
    "setop_all_duplicate_mismatch",
    "json_predicate_order_mismatch",
    "sparse_mask_mismatch",
    "float_wrap_mismatch",
    "index_bool_mismatch",
    "empty_literal_groupby_mismatch",
    "arrow_string_eq_sum_mismatch",
    "arrow_timestamp_loc_slice_mismatch",
    "arrow_timestamp_index_attr_mismatch",
    "eval_inplace_alias_mismatch",
    "bool_reduction_skipna_mismatch",
    "dataset_isin_all_match_mismatch",
    "run_end_null_compute_mismatch",
    "large_string_partition_mismatch",
    "hash_pivot_wider_mismatch",
    "list_flatten_parent_indices_mismatch",
    "rolling_mean_by_null_count_mismatch",
)
BOOLEAN_PROBE_OUTPUT_PREFIX_SET = frozenset(BOOLEAN_PROBE_OUTPUT_PREFIXES)
BOOLEAN_PROBE_OUTPUT_UNDERSCORE_PREFIXES = tuple(f"{prefix}_" for prefix in BOOLEAN_PROBE_OUTPUT_PREFIXES)

IMMUTABLE_VALUE_TYPES = (str, bytes, int, float, bool, type(None))
NUMERIC_ALIAS_PREFIXES = ("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_")
BOOLEAN_ALIAS_PREFIXES = ("any_", "all_")
MUTATION_OPERATOR_ANNEALING_BASE_TEMPERATURE = 0.12
MUTATION_OPERATOR_ANNEALING_MIN_TEMPERATURE = 0.02
MUTATION_OPERATOR_ANNEALING_DECAY = 0.85
MUTATION_PLAN_MAX_STEPS = 4
MUTATION_PLAN_CANDIDATE_WIDTH = 6
MUTATION_PLAN_MAX_ENERGY_CANDIDATES = 12
VALUE_CATALOG_SAMPLE_PROBABILITY = 0.50
IR_REWRITE_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "ir_swap_adjacent",
        "ir_pushdown_filter",
        "ir_pull_filter_above_groupby",
        "ir_wrap_with_window",
        "ir_splice_subtree",
        "ir_fold_redundant_op",
    }
)
SHRINK_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "shrink_drop_tail_op",
        "shrink_fold_redundant_op",
        "shrink_merge_adjacent_filters",
        "shrink_inline_single_use_mutate",
    }
)
OPERATION_LIST_ONLY_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "drop_op",
        "ir_swap_adjacent",
        "ir_pushdown_filter",
        "ir_pull_filter_above_groupby",
        "ir_wrap_with_window",
        "ir_splice_subtree",
        "ir_fold_redundant_op",
        "shrink_drop_tail_op",
        "shrink_fold_redundant_op",
        "shrink_merge_adjacent_filters",
    }
)
TABLE_MUTATING_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "value",
        "nullify_value",
        "duplicate_row",
        "drop_row",
        "shuffle_rows",
        "append_normalized_string_membership",
        "append_left_join_coalesce_membership",
        "append_left_join_case_membership",
        "append_sql_union_coalesce_distinct_topk",
        "append_boolean_membership_case_aggregate",
        "append_left_join_boolean_case_aggregate",
        "append_left_join_boolean_coalesce_case_aggregate",
        "append_boolean_antijoin_case_aggregate",
        "append_left_join_boolean_coalesce_filter_aggregate",
        "append_numeric_text_boolean_antijoin_case_aggregate",
        "append_multi_key_membership_case_aggregate",
    }
)
TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "append_normalized_string_membership",
        "append_left_join_coalesce_membership",
        "append_left_join_case_membership",
        "append_sql_union_coalesce_distinct_topk",
        "append_boolean_membership_case_aggregate",
        "append_left_join_boolean_case_aggregate",
        "append_left_join_boolean_coalesce_case_aggregate",
        "append_boolean_antijoin_case_aggregate",
        "append_left_join_boolean_coalesce_filter_aggregate",
        "append_multi_key_membership_case_aggregate",
    }
)
TABLE_ONLY_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "value",
        "nullify_value",
        "duplicate_row",
        "drop_row",
        "shuffle_rows",
    }
)


@dataclass(slots=True)
class _MutationStateCache:
    schema_by_key: dict[tuple[Any, ...], "MutationSchema"] = field(default_factory=dict)
    context_by_key: dict[tuple[Any, ...], "MutationOperationContext | None"] = field(default_factory=dict)
    table_key_by_identity: dict[int, tuple[Any, ...]] = field(default_factory=dict)
    operation_key_by_identity: dict[int, tuple[Any, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class _ValueCatalogMutationContext:
    descriptor: DisagreementDescriptor | None = None
    entry_scores: Mapping[str, float] | None = None
    collect_usage: bool = False
    used_entries: list[dict[str, Any]] = field(default_factory=list)


_ACTIVE_MUTATION_STATE_CACHE: ContextVar[_MutationStateCache | None] = ContextVar(
    "active_mutation_state_cache",
    default=None,
)
_ACTIVE_VALUE_CATALOG_CONTEXT: ContextVar[_ValueCatalogMutationContext | None] = ContextVar(
    "active_value_catalog_context",
    default=None,
)


@dataclass(slots=True)
class MutationResult:
    case: Case
    metadata: dict[str, Any]


@dataclass(slots=True)
class MutationPlanStep:
    operator: str
    detail: str
    changed: bool
    productive: bool
    heuristic_score: float
    operator_score: float
    target_affinity: float
    divergence_affinity: float
    novelty_bonus: float
    candidate_width: int
    resulting_operation_count: int
    replay_seed: int
    repair_changed: bool = False
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "detail": self.detail,
            "changed": self.changed,
            "productive": self.productive,
            "repair_changed": self.repair_changed,
            "fallback_used": self.fallback_used,
            "heuristic_score": self.heuristic_score,
            "operator_score": self.operator_score,
            "target_affinity": self.target_affinity,
            "divergence_affinity": self.divergence_affinity,
            "novelty_bonus": self.novelty_bonus,
            "candidate_width": self.candidate_width,
            "resulting_operation_count": self.resulting_operation_count,
            "replay_seed": self.replay_seed,
        }


@dataclass(slots=True)
class MutationPlan:
    strategy: str
    planned_depth: int
    executed_depth: int
    steps: list[MutationPlanStep] = field(default_factory=list)
    top_operators: list[dict[str, Any]] = field(default_factory=list)
    annealing_temperature: float = 0.0
    changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "planned_depth": self.planned_depth,
            "executed_depth": self.executed_depth,
            "changed": self.changed,
            "annealing_temperature": self.annealing_temperature,
            "top_operators": [dict(row) for row in self.top_operators],
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True, init=False)
class MutationOperator:
    name: str
    apply: Callable[[list[TableData], list[dict[str, Any]], random.Random], str]
    semantic_family_affinity: tuple[str, ...]
    semantic_signal_affinity: tuple[str, ...]
    exploration_objective_affinity: tuple[str, ...]
    divergence_affinity: tuple[str, ...]

    def __init__(
        self,
        name: str,
        apply: Callable[[list[TableData], list[dict[str, Any]], random.Random], str],
        semantic_family_affinity: tuple[str, ...] = (),
        semantic_signal_affinity: tuple[str, ...] = (),
        exploration_objective_affinity: tuple[str, ...] = (),
        divergence_affinity: tuple[str, ...] = (),
        *,
        semantic_affinity: tuple[str, ...] | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "apply", apply)
        family_affinity = semantic_family_affinity if semantic_affinity is None else semantic_affinity
        object.__setattr__(self, "semantic_family_affinity", tuple(family_affinity))
        object.__setattr__(self, "semantic_signal_affinity", tuple(semantic_signal_affinity))
        object.__setattr__(
            self,
            "exploration_objective_affinity",
            tuple(exploration_objective_affinity),
        )
        object.__setattr__(self, "divergence_affinity", tuple(divergence_affinity))

    @property
    def semantic_affinity(self) -> tuple[str, ...]:
        return self.semantic_family_affinity


@dataclass(frozen=True, slots=True)
class MutationSchema:
    columns: list[str]
    column_types: dict[str, str]
    nullable_columns: set[str]

    @property
    def available(self) -> list[str]:
        return list(self.columns)

    def column_type(self, name: str) -> str:
        if name in self.column_types:
            return self.column_types[name]
        return _fallback_column_type(name)

    def column_has_null(self, name: str) -> bool:
        return name in self.nullable_columns


@dataclass(frozen=True, slots=True)
class MutationOperationContext:
    table: TableData
    schema: MutationSchema
    available: tuple[str, ...]
    numeric: tuple[str, ...]
    bools: tuple[str, ...]
    strings: tuple[str, ...]
    date_strings: tuple[str, ...]
    numeric_strings: tuple[str, ...]
    extra_tables: tuple[TableData, ...] = ()

    def column_type(self, name: str) -> str:
        return self.schema.column_type(name)

    def column_has_null(self, name: str) -> bool:
        return self.schema.column_has_null(name)


@contextmanager
def _activate_mutation_state_cache(cache: _MutationStateCache | None):
    if cache is None:
        yield
        return
    token = _ACTIVE_MUTATION_STATE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_MUTATION_STATE_CACHE.reset(token)


@contextmanager
def _activate_value_catalog_context(context: _ValueCatalogMutationContext | None):
    if context is None:
        yield
        return
    token = _ACTIVE_VALUE_CATALOG_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_VALUE_CATALOG_CONTEXT.reset(token)


def mutate_case(case: Case, seed: int) -> Case:
    return mutate_case_with_metadata(case, seed).case


def mutate_case_with_metadata(
    case: Case,
    seed: int,
    *,
    allow_probe_operators: bool = True,
    operator_scores: Mapping[str, float] | None = None,
    target_keys: Sequence[str] | None = None,
    plan_depth: int | None = None,
    disagreement: DisagreementDescriptor | Mapping[str, Any] | None = None,
    operator_pulls: Mapping[str, int] | None = None,
    recent_operator_pulls: Mapping[str, int] | None = None,
    champion_donors: Sequence[ChampionSeed] | None = None,
    enable_ir_rewrite_mutations: bool = True,
    enable_divergence_conditioned_mutations: bool = True,
    enable_shrink_mutations: bool = True,
    enable_per_operator_energy: bool = True,
    enable_value_catalog: bool = True,
    value_catalog_scores: Mapping[str, float] | None = None,
) -> MutationResult:
    if champion_donors:
        graft_rnd = random.Random(seed ^ 0xC0FFEE)
        if graft_rnd.random() < 0.20:
            donor = champion_donors[0]
            grafted = ChampionRegistry().graft_subtree(case, donor, graft_rnd)
            metadata = _inherited_metadata(case.metadata)
            metadata.update(dict(grafted.metadata or {}))
            metadata.update(
                {
                    "candidate_source": "feedback_mutation",
                    "seed_lineage": {
                        "root_seed": case.seed,
                        "parent_seed": case.seed,
                        "parent_case_id": case.case_id,
                        "mutation_seed": seed,
                        "depth": int((case.metadata or {}).get("seed_lineage", {}).get("depth", 0) or 0) + 1
                        if isinstance(case.metadata, dict)
                        else 1,
                    },
                    "mutation": {
                        "operator": "champion_graft",
                        "detail": f"donor={donor.case_id}",
                        "changed": grafted.program.operations != case.program.operations,
                        "operator_sequence": ["champion_graft"],
                        "detail_sequence": [f"donor={donor.case_id}"],
                        "plan_depth": 1,
                        "executed_steps": 1,
                        "multi_step": False,
                    },
                    "mutation_selection": {
                        "operator": "champion_graft",
                        "strategy": "champion_seed_grafting",
                    },
                    "mutation_plan": {
                        "strategy": "champion_seed_grafting",
                        "planned_depth": 1,
                        "executed_depth": 1,
                        "changed": grafted.program.operations != case.program.operations,
                        "steps": [],
                    },
                }
            )
            grafted.case_id = f"{case.case_id}-champion-mut-{seed}"
            grafted.seed = seed
            grafted.metadata = metadata
            return MutationResult(grafted, metadata)
    operator_pool = _mutation_operator_pool(
        allow_probe_operators=allow_probe_operators,
        enable_ir_rewrite_mutations=enable_ir_rewrite_mutations,
        enable_shrink_mutations=enable_shrink_mutations,
    )
    plan = _build_mutation_plan(
        case,
        seed,
        operator_pool=operator_pool,
        operator_scores=operator_scores,
        target_keys=target_keys,
        plan_depth=plan_depth,
        disagreement=disagreement,
        operator_pulls=operator_pulls,
        recent_operator_pulls=recent_operator_pulls,
        enable_divergence_conditioned_mutations=enable_divergence_conditioned_mutations,
        enable_per_operator_energy=enable_per_operator_energy,
        enable_value_catalog=enable_value_catalog,
        value_catalog_scores=value_catalog_scores,
    )
    if not plan.steps:
        raise RuntimeError("mutation operator pool is empty")
    disagreement_descriptor = _coerce_disagreement_descriptor(disagreement)
    value_catalog_context = (
        _ValueCatalogMutationContext(
            descriptor=disagreement_descriptor,
            entry_scores=value_catalog_scores,
            collect_usage=True,
        )
        if enable_value_catalog
        else None
    )
    with _activate_value_catalog_context(value_catalog_context):
        tables, operations = _apply_mutation_plan(case, seed, operator_pool=operator_pool, plan=plan)
    primary_step = plan.steps[0]
    choice = primary_step.operator
    detail = primary_step.detail
    changed = bool(plan.changed)
    table = tables[0]
    parent_lineage = case.metadata.get("seed_lineage", {}) if isinstance(case.metadata, dict) else {}
    root_seed = parent_lineage.get("root_seed", case.seed)
    depth = int(parent_lineage.get("depth", 0) or 0) + 1
    metadata = _inherited_metadata(case.metadata)
    metadata.update(
        {
            "candidate_source": "feedback_mutation",
            "seed_lineage": {
                "root_seed": root_seed,
                "parent_seed": case.seed,
                "parent_case_id": case.case_id,
                "mutation_seed": seed,
                "depth": depth,
            },
            "mutation": {
                "operator": choice,
                "detail": detail,
                "changed": changed,
                "operator_sequence": [step.operator for step in plan.steps],
                "detail_sequence": [step.detail for step in plan.steps],
                "plan_depth": plan.planned_depth,
                "executed_steps": plan.executed_depth,
                "multi_step": plan.executed_depth > 1,
                "value_catalog_entries": list(value_catalog_context.used_entries)
                if value_catalog_context is not None
                else [],
            },
            "mutation_selection": _mutation_selection_metadata(
                choice,
                [operator for operator in operator_pool if operator.name in {step.operator for step in plan.steps}]
                or list(operator_pool),
                operator_scores=operator_scores,
                plan=plan,
            ),
            "mutation_plan": plan.to_dict(),
        }
    )
    ir_rewrite_rules = _mutation_ir_rewrite_rule_metadata(plan.steps)
    if ir_rewrite_rules:
        metadata["mutation"]["ir_rewrite_rules"] = ir_rewrite_rules
        metadata["mutation"]["ir_rewrite_rule"] = ir_rewrite_rules[0]
    program = Program(
        program_id=f"{case.program.program_id}-mut-{seed}",
        seed=seed,
        operations=operations,
    )
    mutated = Case(
        f"{case.case_id}-mut-{seed}",
        seed,
        tables,
        program,
        metadata=metadata,
    )
    return MutationResult(mutated, metadata)


def _mutation_ir_rewrite_rule_metadata(steps: Sequence[MutationPlanStep]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for step in steps:
        payload = ir_rewrite_rule_metadata(step.operator, detail=step.detail)
        if payload:
            rules.append(payload)
    return rules


def _build_mutation_plan(
    case: Case,
    seed: int,
    *,
    operator_pool: tuple[MutationOperator, ...],
    operator_scores: Mapping[str, float] | None,
    target_keys: Sequence[str] | None,
    plan_depth: int | None,
    disagreement: DisagreementDescriptor | Mapping[str, Any] | None = None,
    operator_pulls: Mapping[str, int] | None = None,
    recent_operator_pulls: Mapping[str, int] | None = None,
    enable_divergence_conditioned_mutations: bool = True,
    enable_per_operator_energy: bool = True,
    enable_value_catalog: bool = True,
    value_catalog_scores: Mapping[str, float] | None = None,
) -> MutationPlan:
    if not operator_pool:
        return MutationPlan(strategy="empty_pool", planned_depth=0, executed_depth=0)
    planned_depth = _resolve_mutation_plan_depth(
        case,
        operator_scores=operator_scores,
        target_keys=target_keys,
        plan_depth=plan_depth,
    )
    operator_score_map = dict(operator_scores or {})
    target_context = _mutation_target_context(target_keys)
    disagreement_descriptor = _coerce_disagreement_descriptor(disagreement)
    original_tables = case.tables
    original_ops = case.program.operations
    current_tables = _clone_tables(case.tables)
    current_operations = _clone_operations(case.program.operations)
    state_cache = _MutationStateCache()
    executed_steps: list[MutationPlanStep] = []
    top_operators: list[dict[str, Any]] = []
    annealing_temperature = 0.0
    for step_index in range(planned_depth):
        step_seed = _mutation_step_seed(seed, case.seed, step_index)
        step_rnd = random.Random(step_seed)
        attempt_order = _mutation_attempt_order(
            operator_pool,
            step_rnd,
            operator_scores=operator_score_map,
        )
        if step_index == 0:
            top_operators = _ranked_operator_snapshot(attempt_order, operator_score_map)
            annealing_temperature = _mutation_selection_temperature(
                attempt_order,
                operator_score_map,
            )
        candidates: list[tuple[list[TableData], list[dict[str, Any]], MutationPlanStep]] = []
        candidate_rank = 1
        candidate_budget = _mutation_plan_candidate_budget(
            operator_pulls=operator_pulls,
            recent_operator_pulls=recent_operator_pulls,
            enable_per_operator_energy=enable_per_operator_energy,
        )
        for operator in attempt_order[: min(len(attempt_order), MUTATION_PLAN_CANDIDATE_WIDTH)]:
            width = _mutation_operator_candidate_width(
                operator,
                operator_scores=operator_score_map,
                operator_pulls=operator_pulls,
                recent_operator_pulls=recent_operator_pulls,
                enable_per_operator_energy=enable_per_operator_energy,
            )
            for _ in range(width):
                if len(candidates) >= candidate_budget:
                    break
                trial_tables, trial_operations, step = _simulate_mutation_step(
                    current_tables,
                    current_operations,
                    operator,
                    seed=seed,
                    case_seed=case.seed,
                    step_index=step_index,
                    candidate_rank=candidate_rank,
                    candidate_width=width,
                    operator_scores=operator_score_map,
                    target_context=target_context,
                    disagreement=disagreement_descriptor,
                    enable_divergence_conditioned_mutations=enable_divergence_conditioned_mutations,
                    enable_value_catalog=enable_value_catalog,
                    value_catalog_scores=value_catalog_scores,
                    prior_steps=executed_steps,
                    state_cache=state_cache,
                )
                candidates.append((trial_tables, trial_operations, step))
                candidate_rank += 1
            if len(candidates) >= candidate_budget:
                break
        if not candidates:
            break
        selected_tables, selected_operations, selected_step = max(
            candidates,
            key=lambda item: (
                item[2].productive,
                item[2].changed,
                item[2].heuristic_score,
                -len(item[1]),
                item[2].operator,
            ),
        )
        if not selected_step.changed:
            if not executed_steps:
                executed_steps.append(selected_step)
            break
        current_tables = selected_tables
        current_operations = selected_operations
        executed_steps.append(selected_step)
        if not selected_step.productive:
            break
        operator_score_map = _advance_plan_operator_scores(
            operator_score_map,
            selected_step.operator,
            executed_steps,
        )
    changed = current_tables != original_tables or current_operations != original_ops
    return MutationPlan(
        strategy="adaptive_multi_step_planning" if planned_depth > 1 else "single_step_ranking",
        planned_depth=planned_depth,
        executed_depth=sum(1 for step in executed_steps if step.changed),
        steps=executed_steps,
        top_operators=top_operators[:8],
        annealing_temperature=annealing_temperature,
        changed=changed,
    )


def _apply_mutation_plan(
    case: Case,
    seed: int,
    *,
    operator_pool: tuple[MutationOperator, ...],
    plan: MutationPlan,
) -> tuple[list[TableData], list[dict[str, Any]]]:
    operator_index = {operator.name: operator for operator in operator_pool}
    tables = _clone_tables(case.tables)
    operations = _clone_operations(case.program.operations)
    state_cache = _MutationStateCache()
    for step_index, step in enumerate(plan.steps):
        operator = operator_index.get(step.operator)
        if operator is None:
            continue
        step_rnd = random.Random(step.replay_seed)
        with _activate_mutation_state_cache(state_cache):
            detail = operator.apply(tables, operations, step_rnd)
        preserves_tables = _operator_preserves_tables(operator)
        preserves_operations = _operator_preserves_operations(operator)
        appends_operations_only = _operator_appends_operations_only(operator)
        mutates_operation_list_only = _operator_mutates_operation_list_only(operator)
        if not preserves_tables:
            _invalidate_table_cache(state_cache, tables)
        if (
            not preserves_operations
            and not appends_operations_only
            and not mutates_operation_list_only
        ):
            _invalidate_operation_cache(state_cache, operations)
        if not preserves_operations:
            operations = repair_operations(tables[0], operations, extra_tables=tables[1:])
            if not operations:
                operations = [{"op": "limit", "n": len(tables[0].rows)}]
        if step.detail != detail:
            step.detail = detail
    return tables, operations


def _simulate_mutation_step(
    current_tables: list[TableData],
    current_operations: list[dict[str, Any]],
    operator: MutationOperator,
    *,
    seed: int,
    case_seed: int,
    step_index: int,
    candidate_rank: int,
    candidate_width: int,
    operator_scores: Mapping[str, float] | None,
    target_context: dict[str, set[str]],
    disagreement: DisagreementDescriptor | None,
    enable_divergence_conditioned_mutations: bool,
    enable_value_catalog: bool,
    value_catalog_scores: Mapping[str, float] | None,
    prior_steps: Sequence[MutationPlanStep],
    state_cache: _MutationStateCache | None = None,
) -> tuple[list[TableData], list[dict[str, Any]], MutationPlanStep]:
    preserves_tables = _operator_preserves_tables(operator)
    preserves_operations = _operator_preserves_operations(operator)
    appends_tables_only = _operator_appends_tables_only(operator)
    appends_operations_only = _operator_appends_operations_only(operator)
    mutates_operation_list_only = _operator_mutates_operation_list_only(operator)
    trial_tables = (
        current_tables
        if preserves_tables
        else list(current_tables)
        if appends_tables_only
        else _clone_primary_table_only(current_tables)
        if preserves_operations
        else _clone_tables(current_tables)
    )
    trial_operations = (
        current_operations
        if preserves_operations
        else list(current_operations)
        if appends_operations_only or mutates_operation_list_only
        else _clone_operations(current_operations)
    )
    replay_seed = _mutation_candidate_seed(seed, case_seed, step_index, candidate_rank, operator.name)
    trial_rnd = random.Random(replay_seed)
    value_context = (
        _ValueCatalogMutationContext(
            descriptor=disagreement,
            entry_scores=value_catalog_scores,
            collect_usage=False,
        )
        if enable_value_catalog
        else None
    )
    with _activate_value_catalog_context(value_context), _activate_mutation_state_cache(state_cache):
        detail = operator.apply(trial_tables, trial_operations, trial_rnd)
    if state_cache is not None and not preserves_tables and not appends_tables_only:
        _invalidate_table_cache(state_cache, trial_tables)
    if (
        state_cache is not None
        and not preserves_operations
        and not appends_operations_only
        and not mutates_operation_list_only
    ):
        _invalidate_operation_cache(state_cache, trial_operations)
    detail_unproductive = _mutation_detail_is_unproductive(detail)
    if preserves_operations:
        changed = trial_tables != current_tables
    elif detail_unproductive and trial_tables == current_tables and trial_operations == current_operations:
        changed = False
    else:
        trial_operations = repair_operations(trial_tables[0], trial_operations, extra_tables=trial_tables[1:])
        if not trial_operations:
            trial_operations = [{"op": "limit", "n": len(trial_tables[0].rows)}]
        changed = trial_tables != current_tables or trial_operations != current_operations
    productive = changed and not detail_unproductive
    operator_score = _mutation_operator_score(operator.name, operator_scores)
    target_affinity = _mutation_target_affinity(operator, target_context)
    divergence_affinity = (
        _mutation_divergence_affinity(operator, disagreement)
        if enable_divergence_conditioned_mutations
        else 0.0
    )
    novelty_bonus = 0.18 if all(step.operator != operator.name for step in prior_steps) else -0.08
    heuristic_score = _mutation_plan_step_score(
        operator=operator,
        detail=detail,
        changed=changed,
        productive=productive,
        operator_score=operator_score,
        target_affinity=target_affinity,
        divergence_affinity=divergence_affinity,
        novelty_bonus=novelty_bonus,
        resulting_operation_count=len(trial_operations),
        prior_steps=prior_steps,
    )
    return trial_tables, trial_operations, MutationPlanStep(
        operator=operator.name,
        detail=detail,
        changed=changed,
        productive=productive,
        heuristic_score=heuristic_score,
        operator_score=operator_score,
        target_affinity=target_affinity,
        divergence_affinity=divergence_affinity,
        novelty_bonus=novelty_bonus + divergence_affinity,
        candidate_width=max(1, int(candidate_width)),
        resulting_operation_count=len(trial_operations),
        replay_seed=replay_seed,
    )


def _operator_preserves_tables(operator: MutationOperator) -> bool:
    return operator.name in MUTATION_OPERATOR_NAMES and operator.name not in TABLE_MUTATING_MUTATION_OPERATOR_NAMES


def _operator_preserves_operations(operator: MutationOperator) -> bool:
    return operator.name in TABLE_ONLY_MUTATION_OPERATOR_NAMES


def _operator_appends_operations_only(operator: MutationOperator) -> bool:
    return operator.name in APPEND_ONLY_MUTATION_OPERATOR_NAMES


def _operator_appends_tables_only(operator: MutationOperator) -> bool:
    return operator.name in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES


def _operator_mutates_operation_list_only(operator: MutationOperator) -> bool:
    return operator.name in OPERATION_LIST_ONLY_MUTATION_OPERATOR_NAMES


def _resolve_mutation_plan_depth(
    case: Case,
    *,
    operator_scores: Mapping[str, float] | None,
    target_keys: Sequence[str] | None,
    plan_depth: int | None,
) -> int:
    if plan_depth is not None:
        return max(1, min(MUTATION_PLAN_MAX_STEPS, int(plan_depth)))
    normalized_targets = [str(target).strip() for target in (target_keys or ()) if str(target).strip()]
    if not normalized_targets:
        return 1
    depth = 1
    lineage = case.metadata.get("seed_lineage", {}) if isinstance(case.metadata, dict) else {}
    current_depth = int(lineage.get("depth", 0) or 0)
    if current_depth <= 1:
        depth += 1
    if len(normalized_targets) >= 2:
        depth += 1
    if any(
        target.startswith("semantic_signal:") or target.startswith("exploration_objective:")
        for target in normalized_targets
    ):
        depth += 1
    if operator_scores:
        positive = [float(score) for key, score in operator_scores.items() if key != "__untried__" and float(score) > 0.0]
        if len(positive) >= 2:
            depth += 1
    return max(1, min(MUTATION_PLAN_MAX_STEPS, depth))


def _mutation_target_context(target_keys: Sequence[str] | None) -> dict[str, set[str]]:
    context = {
        "semantic_family": set(),
        "semantic_signal": set(),
        "exploration_objective": set(),
    }
    for target in target_keys or ():
        text = str(target).strip()
        if not text:
            continue
        if text.startswith("semantic_family:"):
            context["semantic_family"].add(text.removeprefix("semantic_family:").strip())
        elif text.startswith("semantic_signal:"):
            context["semantic_signal"].add(text.removeprefix("semantic_signal:").strip())
        elif text.startswith("exploration_objective:"):
            context["exploration_objective"].add(text.removeprefix("exploration_objective:").strip())
    return context


def _mutation_target_affinity(
    operator: MutationOperator,
    target_context: Mapping[str, set[str]],
) -> float:
    bonus = 0.0
    family_targets = target_context.get("semantic_family", set())
    signal_targets = target_context.get("semantic_signal", set())
    objective_targets = target_context.get("exploration_objective", set())
    if family_targets:
        family_hits = family_targets & set(operator.semantic_family_affinity)
        bonus += 0.18 * len(family_hits)
    if signal_targets:
        signal_hits = signal_targets & set(operator.semantic_signal_affinity)
        bonus += 0.22 * len(signal_hits)
    if objective_targets:
        objective_hits = objective_targets & set(operator.exploration_objective_affinity)
        bonus += 0.16 * len(objective_hits)
    return min(0.75, bonus)


def _coerce_disagreement_descriptor(
    disagreement: DisagreementDescriptor | Mapping[str, Any] | None,
) -> DisagreementDescriptor | None:
    if disagreement is None:
        return None
    if isinstance(disagreement, DisagreementDescriptor):
        return disagreement
    if isinstance(disagreement, Mapping):
        descriptor = DisagreementDescriptor.from_dict(disagreement)
        return descriptor if descriptor.feature_tokens() else None
    return None


def _mutation_divergence_affinity(
    operator: MutationOperator,
    descriptor: DisagreementDescriptor | None,
) -> float:
    if descriptor is None or not operator.divergence_affinity:
        return 0.0
    tokens = set(descriptor.feature_tokens())
    if not tokens:
        return 0.0
    hits = sum(1 for token in operator.divergence_affinity if token in tokens)
    if hits <= 0:
        return 0.0
    return 0.30 * min(1.0, hits / max(1, len(operator.divergence_affinity)))


def _mutation_plan_candidate_budget(
    *,
    operator_pulls: Mapping[str, int] | None,
    recent_operator_pulls: Mapping[str, int] | None,
    enable_per_operator_energy: bool = True,
) -> int:
    if not enable_per_operator_energy or (operator_pulls is None and recent_operator_pulls is None):
        return MUTATION_PLAN_CANDIDATE_WIDTH
    return MUTATION_PLAN_MAX_ENERGY_CANDIDATES


def _mutation_operator_candidate_width(
    operator: MutationOperator,
    *,
    operator_scores: Mapping[str, float],
    operator_pulls: Mapping[str, int] | None,
    recent_operator_pulls: Mapping[str, int] | None,
    enable_per_operator_energy: bool = True,
) -> int:
    if not enable_per_operator_energy or (operator_pulls is None and recent_operator_pulls is None):
        return 1
    return operator_energy(
        pulls=int((operator_pulls or {}).get(operator.name, 0) or 0),
        mean_reward=float(operator_scores.get(operator.name, operator_scores.get("__untried__", 0.0)) or 0.0),
        recent_unproductive_streak=int((recent_operator_pulls or {}).get(operator.name, 0) or 0),
        catalog_width=MUTATION_PLAN_CANDIDATE_WIDTH,
    )


def _mutation_plan_step_score(
    *,
    operator: MutationOperator,
    detail: str,
    changed: bool,
    productive: bool,
    operator_score: float,
    target_affinity: float,
    divergence_affinity: float,
    novelty_bonus: float,
    resulting_operation_count: int,
    prior_steps: Sequence[MutationPlanStep],
) -> float:
    score = operator_score + target_affinity + divergence_affinity + novelty_bonus
    if changed:
        score += 0.85
    else:
        score -= 1.25
    if productive:
        score += 0.65
    else:
        score -= 0.25
    if detail.startswith("append_") or detail.startswith("join:") or detail.startswith("groupby:"):
        score += 0.10
    repeated_penalty = 0.0
    if prior_steps and any(step.operator == operator.name for step in prior_steps):
        repeated_penalty += 0.20
    complexity_penalty = max(0.0, resulting_operation_count - 8) * 0.06
    return score - repeated_penalty - complexity_penalty


def _mutation_step_seed(seed: int, case_seed: int, step_index: int) -> int:
    return (seed * 104729) + (case_seed * 13007) + (step_index * 4099)


def _mutation_candidate_seed(seed: int, case_seed: int, step_index: int, candidate_rank: int, operator_name: str) -> int:
    operator_hash = sum((index + 1) * ord(char) for index, char in enumerate(operator_name))
    return _mutation_step_seed(seed, case_seed, step_index) + (candidate_rank * 811) + operator_hash


def _mutation_operator_score(operator_name: str, operator_scores: Mapping[str, float] | None) -> float:
    if not operator_scores:
        return 0.0
    untried = float(operator_scores.get("__untried__", 0.0))
    return float(operator_scores.get(operator_name, untried))


def _advance_plan_operator_scores(
    operator_scores: Mapping[str, float],
    selected_operator: str,
    prior_steps: Sequence[MutationPlanStep],
) -> dict[str, float]:
    updated = dict(operator_scores)
    if selected_operator:
        updated[selected_operator] = float(updated.get(selected_operator, updated.get("__untried__", 0.0))) - 0.20
    if len(prior_steps) >= 2:
        updated["__untried__"] = float(updated.get("__untried__", 0.0)) + 0.05
    return updated


def _ranked_operator_snapshot(
    attempt_order: Sequence[MutationOperator],
    operator_scores: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "operator": operator.name,
            "score": _mutation_operator_score(operator.name, operator_scores),
        }
        for operator in attempt_order[:8]
    ]


def _mutation_selection_temperature(
    attempt_order: Sequence[MutationOperator],
    operator_scores: Mapping[str, float] | None,
) -> float:
    if not operator_scores:
        return 0.0
    untried_score = float(operator_scores.get("__untried__", 0.0))
    return _mutation_annealing_temperature(
        list(attempt_order[1:]),
        operator_scores=operator_scores,
        untried_score=untried_score,
    )


def _mutation_attempt_order(
    operator_pool: tuple[MutationOperator, ...],
    rnd: random.Random,
    *,
    operator_scores: Mapping[str, float] | None = None,
) -> list[MutationOperator]:
    order = rnd.sample(list(operator_pool), k=len(operator_pool))
    if not operator_scores:
        return order
    untried_score = float(operator_scores.get("__untried__", 0.0))
    ranked = sorted(
        order,
        key=lambda operator: (
            float(operator_scores.get(operator.name, untried_score)),
            rnd.random(),
        ),
        reverse=True,
    )
    if len(ranked) <= 2:
        return ranked
    temperature = _mutation_annealing_temperature(
        ranked[1:],
        operator_scores=operator_scores,
        untried_score=untried_score,
    )
    return [
        ranked[0],
        *_annealed_operator_tail(
            ranked[1:],
            rnd,
            operator_scores=operator_scores,
            untried_score=untried_score,
            temperature=temperature,
        ),
    ]


def _annealed_operator_tail(
    ranked_tail: list[MutationOperator],
    rnd: random.Random,
    *,
    operator_scores: Mapping[str, float],
    untried_score: float,
    temperature: float,
) -> list[MutationOperator]:
    """Diversify near-tied mutation operators without displacing the best arm."""

    remaining = list(ranked_tail)
    selected: list[MutationOperator] = []
    current_temperature = max(0.0, float(temperature))
    while remaining:
        if current_temperature <= 0.0 or len(remaining) == 1:
            selected.extend(remaining)
            break
        best = remaining[0]
        proposal_index = rnd.randrange(len(remaining))
        proposal = remaining[proposal_index]
        best_score = float(operator_scores.get(best.name, untried_score))
        proposal_score = float(operator_scores.get(proposal.name, untried_score))
        delta = proposal_score - best_score
        accept = proposal_index == 0 or delta >= 0.0 or rnd.random() < math.exp(delta / current_temperature)
        selected.append(remaining.pop(proposal_index if accept else 0))
        current_temperature *= MUTATION_OPERATOR_ANNEALING_DECAY
    return selected


def _mutation_annealing_temperature(
    ranked_tail: list[MutationOperator],
    *,
    operator_scores: Mapping[str, float],
    untried_score: float,
) -> float:
    if len(ranked_tail) <= 1:
        return 0.0
    scores = [float(operator_scores.get(operator.name, untried_score)) for operator in ranked_tail]
    spread = max(scores) - min(scores)
    # Wider score separation means feedback is confident; near-ties keep more exploration.
    return max(
        MUTATION_OPERATOR_ANNEALING_MIN_TEMPERATURE,
        MUTATION_OPERATOR_ANNEALING_BASE_TEMPERATURE / (1.0 + max(0.0, spread)),
    )


def _mutation_selection_metadata(
    selected_operator: str,
    attempt_order: list[MutationOperator],
    *,
    operator_scores: Mapping[str, float] | None,
    plan: MutationPlan | None = None,
) -> dict[str, Any]:
    if not operator_scores:
        payload = {
            "strategy": "random_operator_shuffle",
            "candidate_count": len(attempt_order),
            "selected_operator": selected_operator,
        }
        if plan is not None:
            payload["planned_depth"] = plan.planned_depth
            payload["executed_depth"] = plan.executed_depth
            payload["multi_step"] = plan.executed_depth > 1
            payload["plan"] = plan.to_dict()
        return payload
    untried_score = float(operator_scores.get("__untried__", 0.0))
    scored = [
        {
            "operator": operator.name,
            "score": float(operator_scores.get(operator.name, untried_score)),
        }
        for operator in attempt_order
    ]
    ranked = sorted(scored, key=lambda row: (row["score"], row["operator"]), reverse=True)
    selected_score = float(operator_scores.get(selected_operator, untried_score))
    selected_rank = next(
        (index + 1 for index, row in enumerate(ranked) if row["operator"] == selected_operator),
        0,
    )
    tail = [operator for operator in attempt_order if operator.name != ranked[0]["operator"]]
    payload = {
        "strategy": "feedback_score_with_annealed_tail",
        "candidate_count": len(attempt_order),
        "selected_operator": selected_operator,
        "selected_operator_score": selected_score,
        "selected_score_rank": selected_rank,
        "annealing_temperature": _mutation_annealing_temperature(
            tail,
            operator_scores=operator_scores,
            untried_score=untried_score,
        ),
        "annealing_decay": MUTATION_OPERATOR_ANNEALING_DECAY,
        "top_operators": plan.top_operators[:8] if plan is not None and plan.top_operators else ranked[:8],
    }
    if plan is not None:
        payload["planned_depth"] = plan.planned_depth
        payload["executed_depth"] = plan.executed_depth
        payload["multi_step"] = plan.executed_depth > 1
        payload["plan"] = plan.to_dict()
        if plan.annealing_temperature > 0.0:
            payload["annealing_temperature"] = plan.annealing_temperature
    return payload


def mutation_operator_profiles(
    *,
    allow_probe_operators: bool = True,
    enable_ir_rewrite_mutations: bool = True,
    enable_shrink_mutations: bool = True,
) -> Mapping[str, MutationOperator]:
    profiles = ALL_MUTATION_OPERATOR_PROFILES if allow_probe_operators else DISCOVERY_MUTATION_OPERATOR_PROFILES
    disabled_names: set[str] = set()
    if not enable_ir_rewrite_mutations:
        disabled_names.update(IR_REWRITE_MUTATION_OPERATOR_NAMES)
    if not enable_shrink_mutations:
        disabled_names.update(SHRINK_MUTATION_OPERATOR_NAMES)
    if not disabled_names:
        return profiles
    return MappingProxyType(
        {
            name: operator
            for name, operator in profiles.items()
            if name not in disabled_names
        }
    )


def _mutation_operator_pool(
    *,
    allow_probe_operators: bool,
    enable_ir_rewrite_mutations: bool = True,
    enable_shrink_mutations: bool = True,
) -> tuple[MutationOperator, ...]:
    pool = MUTATION_OPERATORS if allow_probe_operators else DISCOVERY_MUTATION_OPERATORS
    disabled_names: set[str] = set()
    if not enable_ir_rewrite_mutations:
        disabled_names.update(IR_REWRITE_MUTATION_OPERATOR_NAMES)
    if not enable_shrink_mutations:
        disabled_names.update(SHRINK_MUTATION_OPERATOR_NAMES)
    if not disabled_names:
        return pool
    return tuple(operator for operator in pool if operator.name not in disabled_names)


def _mutation_detail_is_unproductive(detail: str) -> bool:
    tokens = detail.split(":")[1:]
    if not tokens:
        return False
    first = tokens[0]
    return (
        first == "none"
        or first.startswith("no-")
        or first.startswith("not-enough")
        or first in {"too-few-columns", "duplicate-keys"}
    )


def _mutation_schema(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    *,
    cache_key: tuple[Any, ...] | None = None,
) -> MutationSchema:
    cache = _ACTIVE_MUTATION_STATE_CACHE.get()
    if cache is not None and cache_key is None:
        cache_key = _mutation_state_cache_key(tables, operations, cache=cache)
    if cache is not None and cache_key is not None:
        cached = cache.schema_by_key.get(cache_key)
        if cached is not None:
            return cached
    if not tables:
        schema = MutationSchema([], {}, set())
        if cache is not None and cache_key is not None:
            cache.schema_by_key[cache_key] = schema
        return schema
    state = state_after_operations(
        tables[0],
        operations,
        extra_tables=tables[1:],
    )
    nullable_columns = set(state.nullable_columns)
    base_null_columns = _base_table_null_columns(tables)
    nullable_columns.update(name for name in state.columns if name in base_null_columns)
    schema = MutationSchema(list(state.columns), dict(state.column_types), nullable_columns)
    if cache is not None and cache_key is not None:
        cache.schema_by_key[cache_key] = schema
    return schema


def _base_table_column_has_null(tables: list[TableData], name: str) -> bool:
    return any(name in row and row.get(name) is None for table in tables for row in table.rows)


def _base_table_null_columns(tables: list[TableData]) -> set[str]:
    null_columns: set[str] = set()
    for table in tables:
        null_columns.update(_table_null_columns(table))
    return null_columns


def _fallback_column_type(name: str) -> str:
    if name.startswith("sorted_ok_"):
        return "bool"
    if name in BOOLEAN_PROBE_OUTPUT_PREFIX_SET or name.startswith(BOOLEAN_PROBE_OUTPUT_UNDERSCORE_PREFIXES):
        return "bool"
    if name.startswith(("any_", "all_")):
        return "bool"
    return "float" if name.startswith(("m_", "sum_", "min_", "max_", "run_")) else "int"


def _inherited_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {key: _clone_value(metadata[key]) for key in INHERITED_METADATA_KEYS if key in metadata}


def _clone_tables(tables: list[TableData]) -> list[TableData]:
    return [_clone_table(table) for table in tables]


def _clone_primary_table_only(tables: list[TableData]) -> list[TableData]:
    if not tables:
        return []
    return [_clone_table(tables[0]), *tables[1:]]


def _clone_table(table: TableData) -> TableData:
    return TableData(
        table.name,
        [ColumnSpec(column.name, column.type, nullable=column.nullable) for column in table.columns],
        [_clone_row(row) for row in table.rows],
    )


def _clone_row(row: dict[str, Any]) -> dict[str, Any]:
    for value in row.values():
        if not isinstance(value, IMMUTABLE_VALUE_TYPES):
            return {key: _clone_value(item) for key, item in row.items()}
    return dict(row)


def _clone_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clone_operation(operation) for operation in operations]


def _clone_operation(operation: dict[str, Any]) -> dict[str, Any]:
    if isinstance(operation, IRNode):
        return operation.to_dict()
    return {
        key: _clone_value(value)
        for key, value in operation.items()
    }


def _clone_value(value: Any) -> Any:
    if isinstance(value, IMMUTABLE_VALUE_TYPES):
        return value
    if isinstance(value, IRNode):
        return value.copy()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return {_clone_value(item) for item in value}
    return copy.deepcopy(value)


def _mutate_scalar_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables:
        return "value:none"
    return _mutate_value(tables[0], rnd)


def _mutate_value(table: TableData, rnd: random.Random) -> str:
    if not table.rows:
        return "value:none"
    col = rnd.choice(table.columns)
    row = rnd.choice(table.rows)
    current = row.get(col.name)
    catalog = _sample_catalog_literal(col.type, rnd, column=col.name)
    if current is None:
        if catalog is not None:
            row[col.name] = catalog.value
            _record_value_catalog_usage(catalog, column=col.name, column_type=col.type, purpose="fill-null")
            return f"value_catalog:{catalog.entry_id}:{col.name}:fill-null"
        if col.type == "str" and _looks_like_date_column(col.name):
            row[col.name] = rnd.choice(["2024-01-03", "2024-02-14T08:30:00", "2025-12-31", "2026-01-01"])
        else:
            row[col.name] = _literal_for_type(col.type, rnd)
        return f"value:{col.type}:fill-null"
    if catalog is not None and rnd.random() < 0.70:
        row[col.name] = catalog.value
        _record_value_catalog_usage(catalog, column=col.name, column_type=col.type, purpose="replace")
        return f"value_catalog:{catalog.entry_id}:{col.name}"
    if col.type == "int":
        row[col.name] = int(current) + rnd.choice([-10, -1, 0, 1, 10])
    elif col.type == "float":
        if isinstance(current, float) and (math.isnan(current) or math.isinf(current)):
            row[col.name] = 0.0
        else:
            row[col.name] = float(current) + rnd.choice([-1.0, -0.5, 0.5, 1.0])
    elif col.type == "bool":
        row[col.name] = not bool(current)
    elif col.type == "str":
        if _looks_like_date_column(col.name):
            row[col.name] = rnd.choice(["2024-01-03", "2024-02-14T08:30:00", "2025-12-31", "2026-01-01"])
        else:
            row[col.name] = rnd.choice(["", "alpha", "ALPHA", "中文", str(current) + "_x"])
    return f"value:{col.type}:{col.name}"


def _nullify_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "nullify:none"
    table = tables[0]
    nullable = [col for col in table.columns if col.nullable]
    if not nullable:
        return "nullify:no-nullable-column"
    candidates = [
        (row, col)
        for row in table.rows
        for col in nullable
        if row.get(col.name) is not None
    ]
    if not candidates:
        candidates = [(row, col) for row in table.rows for col in nullable]
    row, col = rnd.choice(candidates)
    row[col.name] = None
    return f"nullify:{col.type}:{col.name}"


def _duplicate_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "duplicate_row:none"
    tables[0].rows.append(_clone_row(rnd.choice(tables[0].rows)))
    return f"duplicate_row:{tables[0].name}"


def _drop_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "drop_row:none"
    removed_index = rnd.randrange(len(tables[0].rows))
    tables[0].rows.pop(removed_index)
    return f"drop_row:{tables[0].name}:{removed_index}"


def _shuffle_rows(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or len(tables[0].rows) < 2:
        return "shuffle_rows:none"
    before = list(tables[0].rows)
    rnd.shuffle(tables[0].rows)
    if tables[0].rows == before:
        tables[0].rows.reverse()
    return f"shuffle_rows:{tables[0].name}"


def _append_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    op = _random_operation(tables, operations, rnd)
    if op is not None:
        operations.append(op)
        return f"append:{op_kind(op, 'unknown')}"
    return "append:none"


def _drop_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del tables
    if len(operations) <= 1:
        return "drop:none"
    removed = operations.pop(rnd.randrange(len(operations)))
    return f"drop:{op_kind(removed, 'unknown')}"


def _tweak_random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not operations:
        return "tweak:none"
    op = operations[rnd.randrange(len(operations))]
    return _tweak_operation(tables, op, rnd)


def _append_order_projection_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_order_projection:none"
    available = _available_columns(tables, operations)
    if len(available) < 2:
        return "append_order_projection:too-few-columns"
    primary = rnd.choice(available)
    selected_candidates = [column for column in available if column != primary]
    selected_count = rnd.randint(1, min(3, len(selected_candidates)))
    selected = sorted(rnd.sample(selected_candidates, selected_count))
    sort_columns = [primary] + sorted(column for column in available if column != primary)
    operations.append(
        {
            "op": "sort",
            "keys": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in sort_columns
            ],
        }
    )
    operations.append({"op": "select", "columns": selected})
    if rnd.random() < 0.75:
        if rnd.random() < 0.70:
            operations.append({"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))})
            tail = "limit"
        else:
            operations.append({"op": "offset", "n": rnd.randint(0, 2)})
            tail = "offset"
    else:
        tail = "none"
    return f"append_order_projection:{primary}:tail={tail}"


def _append_truth_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_truth_filter:none"
    available = _available_columns(tables, operations)
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric:
        return "append_truth_filter:no-numeric-column"
    column = rnd.choice(numeric)
    comparator = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
    operations.append(
        {
            "op": "filter",
            "column": column,
            "cmp": comparator,
            "value": _literal_for_type(_column_type(tables, column), rnd),
        }
    )
    return f"append_truth_filter:{column}:{comparator}"


def _append_boolean_predicate_filter_probe(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_predicate_filter:none"
    available = _available_columns(tables, operations)
    boolean_columns = [column for column in available if _column_type(tables, column) == "bool"]
    if not boolean_columns:
        return "append_boolean_predicate_filter:no-bool-column"
    column = rnd.choice(boolean_columns)
    comparator = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
    operations.append({"op": "filter", "column": column, "cmp": comparator, "value": None})
    return f"append_boolean_predicate_filter:{column}:{comparator}"


def _append_range_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_range_filter:none"
    available = _available_columns(tables, operations)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric_columns:
        return "append_range_filter:no-numeric-column"
    column = rnd.choice(numeric_columns)
    values = sorted(rnd.sample(_literal_list_for_type(_column_type(tables, column), rnd), 2))
    operations.append({"op": "filter", "column": column, "cmp": "range_closed", "value": values})
    return f"append_range_filter:{column}:{values[0]}:{values[1]}"


def _append_tuple_absence_filter_probe(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if len(tables) < 2:
        return "append_tuple_absence_filter:no-right-table"
    available = _available_columns(tables, operations)
    right = rnd.choice(tables[1:])
    left_types = {column: _column_type(tables, column) for column in available}
    right_types = {column.name: column.type for column in right.columns}
    pairs = [
        (left, right_column)
        for left, left_type in left_types.items()
        for right_column, right_type in right_types.items()
        if left_type == right_type
    ]
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for left, right_column in rnd.sample(pairs, k=len(pairs)):
        if left in left_seen or right_column in right_seen:
            continue
        chosen.append((left, right_column))
        left_seen.add(left)
        right_seen.add(right_column)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return "append_tuple_absence_filter:no-compatible-pairs"
    left_columns = [left for left, _ in chosen]
    selected_right_columns = [right_column for _, right_column in chosen]
    operations.append(
        {
            "op": "tuple_absence_filter",
            "columns": left_columns,
            "table": right.name,
            "right_columns": selected_right_columns,
        }
    )
    return f"append_tuple_absence_filter:{','.join(left_columns)}:{right.name}:{','.join(selected_right_columns)}"


def _append_row_value_absence_filter(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    detail = _append_tuple_absence_filter_probe(tables, operations, rnd)
    if detail.startswith("append_tuple_absence_filter:"):
        return "append_row_value_absence_filter:" + detail.split(":", 1)[1]
    return detail.replace("append_tuple_absence_filter", "append_row_value_absence_filter", 1)


def _append_running_sum_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_running_sum:none"
    available = _available_columns(tables, operations)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric_columns or not available:
        return "append_running_sum:no-numeric-column"
    source = rnd.choice(numeric_columns)
    order_column = rnd.choice(available)
    order_columns = [order_column] + sorted(column for column in available if column != order_column)
    used = set(available)
    output_column = make_safe_output_name(f"run_{source}", used=used)
    operations.append(
        {
            "op": "running_sum",
            "source": source,
            "column": output_column,
            "order_by": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in order_columns
            ],
            "input_dtype": "float32" if _column_type(tables, source) == "float" or rnd.random() < 0.5 else "float64",
        }
    )
    return f"append_running_sum:{source}:order={','.join(order_columns)}:out={output_column}"


def _append_sortedness_check_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_sortedness_check:none"
    available = _available_columns(tables, operations)
    candidates = [
        column
        for column in available
        if _column_type(tables, column) in {"int", "float", "str", "bool"}
    ]
    if not candidates:
        return "append_sortedness_check:no-comparable-column"
    null_candidates = [column for column in candidates if _column_has_null(tables, column)]
    column = rnd.choice(null_candidates or candidates)
    sort_nulls = rnd.choice(["first", "last"])
    check_nulls = "last" if sort_nulls == "first" else "first"
    ascending = rnd.choice([True, False])
    alias = make_safe_output_name(f"sorted_ok_{column}", used=set(available))
    operations.extend(
        [
            {
                "op": "sort",
                "keys": [{"column": column, "ascending": ascending, "nulls": sort_nulls}],
            },
            {
                "op": "sortedness_check",
                "column": column,
                "as": alias,
                "ascending": ascending,
                "nulls": check_nulls,
            },
        ]
    )
    return f"append_sortedness_check:{column}:sort_nulls={sort_nulls}:check_nulls={check_nulls}:out={alias}"


def _append_random_case_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_random_case_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("unexpected_else_seen", used=set(available))
    rows = rnd.choice([50_000, 100_000, 150_000])
    branches = rnd.choice([3, 4])
    operations.append({"op": "random_case_probe", "as": alias, "rows": rows, "branches": branches})
    return f"append_random_case_probe:rows={rows}:branches={branches}:out={alias}"


def _append_group_quantile_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_group_quantile_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("quantile_key_mismatch", used=set(available))
    operations.append(
        {
            "op": "group_quantile_probe",
            "as": alias,
            "values": [1, 2, 3],
            "quantiles": [0.0, 0.5, 1.0],
        }
    )
    return f"append_group_quantile_probe:out={alias}"


def _append_scalar_subquery_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_scalar_subquery_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("scalar_subquery_mismatch", used=set(available))
    operations.append({"op": "scalar_subquery_probe", "as": alias})
    return f"append_scalar_subquery_probe:out={alias}"


def _append_window_avg_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_window_avg_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("window_avg_mismatch", used=set(available))
    operations.append({"op": "window_avg_probe", "as": alias})
    return f"append_window_avg_probe:out={alias}"


def _append_struct_distinct_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_struct_distinct_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("struct_distinct_mismatch", used=set(available))
    operations.append({"op": "struct_distinct_probe", "as": alias})
    return f"append_struct_distinct_probe:out={alias}"


def _append_bit_compare_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_bit_compare_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("bit_compare_mismatch", used=set(available))
    operations.append({"op": "bit_compare_probe", "as": alias})
    return f"append_bit_compare_probe:out={alias}"


def _append_round_even_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_round_even_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("round_even_mismatch", used=set(available))
    operations.append({"op": "round_even_probe", "as": alias})
    return f"append_round_even_probe:out={alias}"


def _append_series_rtruediv_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_series_rtruediv_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("series_rtruediv_mismatch", used=set(available))
    operations.append({"op": "series_rtruediv_probe", "as": alias})
    return f"append_series_rtruediv_probe:out={alias}"


def _append_uint64_isin_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_uint64_isin_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("uint64_isin_mismatch", used=set(available))
    operations.append({"op": "uint64_isin_probe", "as": alias})
    return f"append_uint64_isin_probe:out={alias}"


def _append_tuple_anti_null_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_tuple_anti_null_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("tuple_anti_null_mismatch", used=set(available))
    operations.append({"op": "tuple_anti_null_probe", "as": alias})
    return f"append_tuple_anti_null_probe:out={alias}"


def _append_setop_all_duplicate_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_setop_all_duplicate_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("setop_all_duplicate_mismatch", used=set(available))
    operations.append({"op": "setop_all_duplicate_probe", "as": alias})
    return f"append_setop_all_duplicate_probe:out={alias}"


def _append_json_predicate_order_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_json_predicate_order_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("json_predicate_order_mismatch", used=set(available))
    operations.append({"op": "json_predicate_order_probe", "as": alias})
    return f"append_json_predicate_order_probe:out={alias}"


def _append_sparse_mask_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_sparse_mask_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("sparse_mask_mismatch", used=set(available))
    operations.append({"op": "sparse_mask_probe", "as": alias})
    return f"append_sparse_mask_probe:out={alias}"


def _append_float_wrap_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_float_wrap_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("float_wrap_mismatch", used=set(available))
    operations.append({"op": "float_wrap_probe", "as": alias})
    return f"append_float_wrap_probe:out={alias}"


def _append_index_bool_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_index_bool_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("index_bool_mismatch", used=set(available))
    operations.append({"op": "index_bool_probe", "as": alias})
    return f"append_index_bool_probe:out={alias}"


def _append_empty_literal_groupby_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_empty_literal_groupby_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("empty_literal_groupby_mismatch", used=set(available))
    operations.append({"op": "empty_literal_groupby_probe", "as": alias})
    return f"append_empty_literal_groupby_probe:out={alias}"


def _append_arrow_string_eq_sum_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_string_eq_sum_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_string_eq_sum_mismatch", used=set(available))
    operations.append({"op": "arrow_string_eq_sum_probe", "as": alias})
    return f"append_arrow_string_eq_sum_probe:out={alias}"


def _append_arrow_timestamp_loc_slice_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_timestamp_loc_slice_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_timestamp_loc_slice_mismatch", used=set(available))
    operations.append({"op": "arrow_timestamp_loc_slice_probe", "as": alias})
    return f"append_arrow_timestamp_loc_slice_probe:out={alias}"


def _append_arrow_timestamp_index_attr_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_timestamp_index_attr_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_timestamp_index_attr_mismatch", used=set(available))
    operations.append({"op": "arrow_timestamp_index_attr_probe", "as": alias})
    return f"append_arrow_timestamp_index_attr_probe:out={alias}"


def _append_eval_inplace_alias_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_eval_inplace_alias_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("eval_inplace_alias_mismatch", used=set(available))
    operations.append({"op": "eval_inplace_alias_probe", "as": alias})
    return f"append_eval_inplace_alias_probe:out={alias}"


def _append_bool_reduction_skipna_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_bool_reduction_skipna_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("bool_reduction_skipna_mismatch", used=set(available))
    operations.append({"op": "bool_reduction_skipna_probe", "as": alias})
    return f"append_bool_reduction_skipna_probe:out={alias}"


def _append_dataset_isin_all_match_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_dataset_isin_all_match_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("dataset_isin_all_match_mismatch", used=set(available))
    operations.append({"op": "dataset_isin_all_match_probe", "as": alias})
    return f"append_dataset_isin_all_match_probe:out={alias}"


def _append_run_end_null_compute_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_run_end_null_compute_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("run_end_null_compute_mismatch", used=set(available))
    operations.append({"op": "run_end_null_compute_probe", "as": alias})
    return f"append_run_end_null_compute_probe:out={alias}"


def _append_large_string_partition_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_large_string_partition_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("large_string_partition_mismatch", used=set(available))
    operations.append({"op": "large_string_partition_probe", "as": alias})
    return f"append_large_string_partition_probe:out={alias}"


def _append_hash_pivot_wider_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_hash_pivot_wider_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("hash_pivot_wider_mismatch", used=set(available))
    operations.append({"op": "hash_pivot_wider_probe", "as": alias})
    return f"append_hash_pivot_wider_probe:out={alias}"


def _append_list_flatten_parent_indices_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_list_flatten_parent_indices_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("list_flatten_parent_indices_mismatch", used=set(available))
    operations.append({"op": "list_flatten_parent_indices_probe", "as": alias})
    return f"append_list_flatten_parent_indices_probe:out={alias}"


def _append_rolling_mean_by_null_count_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_rolling_mean_by_null_count_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("rolling_mean_by_null_count_mismatch", used=set(available))
    operations.append({"op": "rolling_mean_by_null_count_probe", "as": alias})
    return f"append_rolling_mean_by_null_count_probe:out={alias}"


def _append_grouped_topk_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_grouped_topk:none"
    available = _available_columns(tables, operations)
    if not available:
        return "append_grouped_topk:no-columns"
    numeric = [
        column
        for column in available
        if _column_type(tables, column) in {"int", "float"}
        or column.startswith(("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_"))
    ]
    if not numeric:
        return "append_grouped_topk:no-numeric-column"
    key = rnd.choice(numeric)
    value = rnd.choice([column for column in numeric if column != key] or numeric)
    alias = make_safe_output_name(f"count_{value}", used={key})
    operations.extend(
        [
            {
                "op": "groupby",
                "keys": [key],
                "aggs": [{"column": value, "func": "count", "as": alias}],
            },
            {"op": "select", "columns": [key]},
            {
                "op": "sort",
                "keys": [
                    {
                        "column": key,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                ],
            },
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_grouped_topk:{key}:{value}"


def _append_groupby_fractional_membership_filter(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_groupby_fractional_membership_filter:none"
    available = set(_available_columns(tables, operations))
    candidates: list[tuple[str, str]] = []
    for op in operations:
        if op_kind(op) != "groupby":
            continue
        for agg in aggregate_specs(op):
            alias = aggregate_alias(agg)
            source = aggregate_column(agg)
            if (
                alias in available
                and aggregate_func(agg) in {"min", "max"}
                and _column_type(tables, source) == "int"
            ):
                candidates.append((alias, source))
    if not candidates:
        return "append_groupby_fractional_membership_filter:no-int-aggregate"
    alias, source = rnd.choice(candidates)
    source_values = [
        int(row[source])
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), int) and not isinstance(row.get(source), bool)
    ]
    non_negative_values = [value for value in unique_preserve_order(source_values) if value >= 0]
    if not non_negative_values:
        return "append_groupby_fractional_membership_filter:no-nonnegative-value"
    target = rnd.choice(non_negative_values)
    operations.append(
        {
            "op": "filter",
            "column": alias,
            "cmp": "in_set",
            "value": [float(target) + 0.5, -999.0, 999.0],
        }
    )
    return f"append_groupby_fractional_membership_filter:{alias}:{target}"


def _append_normalized_string_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_normalized_string_membership:none"
    available = _available_columns(tables, operations)
    string_columns = [column for column in available if _column_type(tables, column) == "str"]
    if not string_columns:
        return "append_normalized_string_membership:no-string-column"
    source = rnd.choice(string_columns)
    normalized_values = unique_preserve_order(
        value.strip().lower()
        for table in tables
        for row in table.rows
        for value in [row.get(source)]
        if isinstance(value, str)
    )
    if not normalized_values:
        return f"append_normalized_string_membership:no-values:{source}"
    if len(normalized_values) == 1:
        selected_values = normalized_values
        join_kind = "semi_join"
    else:
        width = rnd.randint(1, max(1, len(normalized_values) - 1))
        selected_values = sorted(rnd.sample(normalized_values, width))
        join_kind = rnd.choice(["semi_join", "anti_join"])
    table_name = make_safe_output_name("t_string_membership_mut", used={table.name for table in tables})
    tables.append(
        TableData(
            table_name,
            [ColumnSpec("s_key", "str", nullable=False)],
            [{"s_key": value} for value in selected_values],
        )
    )
    used_columns = set(available)
    clean_col = make_safe_output_name(f"{source}_clean", used=used_columns)
    used_columns.add(clean_col)
    key_col = make_safe_output_name(f"{source}_key", used=used_columns)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    operations.extend(
        [
            {"op": "mutate", "column": clean_col, "expr": {"kind": "string_strip", "source": source}},
            {"op": "mutate", "column": key_col, "expr": {"kind": "string_lower", "source": clean_col}},
            {"op": "filter", "column": key_col, "cmp": "is_not_null", "value": None},
            {"op": join_kind, "table": table_name, "left_on": key_col, "right_on": "s_key"},
        ]
    )
    if join_kind == "anti_join":
        count_source = numeric_columns[0] if numeric_columns else source
        operations.extend(
            [
                {
                    "op": "groupby",
                    "keys": [key_col],
                    "aggs": [{"column": count_source, "func": "count", "as": "count_rows"}],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": key_col, "ascending": True, "nulls": "last"},
                        {"column": "count_rows", "ascending": False, "nulls": "last"},
                    ],
                },
            ]
        )
    else:
        sort_keys = [{"column": key_col, "ascending": True, "nulls": "last"}]
        if numeric_columns:
            sort_keys.append({"column": numeric_columns[0], "ascending": False, "nulls": "last"})
        selected = unique_preserve_order([key_col, source, *numeric_columns[:1]])
        operations.extend(
            [
                {"op": "sort", "keys": sort_keys},
                {"op": "select", "columns": selected},
                {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
            ]
        )
    return f"append_normalized_string_membership:{join_kind}:{source}:{','.join(map(str, selected_values))}"


def _append_sql_distinct_null_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_sql_distinct_null_topk:none"
    available = _available_columns(tables, operations)
    strings = [column for column in available if _column_type(tables, column) == "str"]
    if not strings:
        return "append_sql_distinct_null_topk:no-string-column"
    used_columns = set(available)
    source = rnd.choice(strings)
    nonempty_col = make_safe_output_name(f"{source}_nonempty", used=used_columns)
    used_columns.add(nonempty_col)
    label_col = nonempty_col
    operations.append(
        {"op": "mutate", "column": nonempty_col, "expr": {"kind": "string_null_if_empty", "source": source}}
    )
    coalesce_sources = [column for column in strings if column != source]
    if coalesce_sources:
        label_col = make_safe_output_name(f"{source}_label", used=used_columns)
        used_columns.add(label_col)
        operations.append(
            {
                "op": "coalesce",
                "columns": [nonempty_col, rnd.choice(coalesce_sources)],
                "as": label_col,
                "fallback": "missing",
            }
        )
    distinct_cols = [label_col]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    distinct_cols.extend((bools + numeric)[:2])
    distinct_cols = unique_preserve_order(distinct_cols)
    sort_keys = [
        {"column": distinct_cols[0], "ascending": True, "nulls": "first"},
        *[
            {
                "column": column,
                "ascending": False if _column_type(tables, column) != "str" else True,
                "nulls": "first" if idx == 0 else "last",
            }
            for idx, column in enumerate(distinct_cols[1:])
        ],
    ]
    operations.extend(
        [
            {"op": "distinct", "columns": distinct_cols},
            {"op": "sort", "keys": sort_keys},
            {"op": "offset", "n": rnd.randint(0, 1)},
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_sql_distinct_null_topk:{source}:label={label_col}:cols={','.join(distinct_cols)}"


def _append_left_join_coalesce_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_coalesce_membership:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    string_candidates = [column for column in available if _column_type(tables, column) == "str"]
    if not key_candidates or not string_candidates:
        return "append_left_join_coalesce_membership:no-key-or-string-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    key_type = base_columns[key].type
    key_values = unique_preserve_order(
        row.get(key)
        for row in tables[0].rows
        if row.get(key) is not None
    )
    if not key_values:
        return f"append_left_join_coalesce_membership:no-key-values:{key}"
    used_table_names = {table.name for table in tables}
    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_membership_mut", used=used_table_names)
    membership_table = make_safe_output_name("t_segment_membership_mut", used=used_table_names | {lookup_table})
    right_key = "lookup_key"
    label_col = make_safe_output_name("lookup_label", used=used_columns)
    used_columns.add(label_col)
    value_col = make_safe_output_name("lookup_j", used=used_columns)
    used_columns.add(value_col)
    segment_col = make_safe_output_name("segment_key", used=used_columns)
    labels = ["dim-a", "dim-b", "space value", None]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, key_type, nullable=False),
                ColumnSpec(label_col, "str", nullable=True),
                ColumnSpec(value_col, "int", nullable=True),
            ],
            [
                {
                    right_key: value,
                    label_col: labels[idx % len(labels)],
                    value_col: None if idx % 4 == 0 else rnd.choice([-2, 0, 1, 3, 8]),
                }
                for idx, value in enumerate(key_values[:8])
            ],
        )
    )
    tables.append(
        TableData(
            membership_table,
            [ColumnSpec(segment_col, "str", nullable=False)],
            [{"segment_key": value} if segment_col == "segment_key" else {segment_col: value} for value in labels if value],
        )
    )
    source_string = rnd.choice(string_candidates)
    join_kind = rnd.choice(["semi_join", "anti_join"])
    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [label_col, source_string], "as": segment_col, "fallback": "missing"},
            {"op": "fill_null", "column": value_col, "value": 0},
            {"op": join_kind, "table": membership_table, "left_on": segment_col, "right_on": segment_col},
        ]
    )
    if join_kind == "semi_join":
        operations.extend(
            [
                {
                    "op": "groupby",
                    "keys": [segment_col],
                    "aggs": [
                        {"column": key, "func": "count", "as": "count_key"},
                        {"column": value_col, "func": "sum", "as": "sum_lookup_j"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": segment_col, "ascending": True, "nulls": "last"},
                        {"column": "count_key", "ascending": False, "nulls": "last"},
                    ],
                },
            ]
        )
    else:
        operations.extend(
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": segment_col, "ascending": True, "nulls": "first"},
                        {"column": value_col, "ascending": False, "nulls": "last"},
                        {"column": key, "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": [key, segment_col, value_col]},
                {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
            ]
        )
    return f"append_left_join_coalesce_membership:{join_kind}:{key}:segment={segment_col}"


def _append_left_join_case_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_case_membership:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    if not key_candidates:
        return "append_left_join_case_membership:no-key-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    key_type = base_columns[key].type
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_case_membership:no-key-values:{key}"
    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_case_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    tag_col = make_safe_output_name("lookup_tag", used=used_columns)
    used_columns.add(tag_col)
    value_col = make_safe_output_name("lookup_j", used=used_columns)
    used_columns.add(value_col)
    bucket_col = make_safe_output_name("tag_bucket", used=used_columns)
    used_columns.add(bucket_col)
    labels = ["dim-a", "dim-b", "space value", None]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, key_type, nullable=False),
                ColumnSpec(tag_col, "str", nullable=True),
                ColumnSpec(value_col, "int", nullable=True),
            ],
            [
                {
                    right_key: value,
                    tag_col: labels[idx % len(labels)],
                    value_col: None if idx % 4 == 0 else rnd.choice([-2, 0, 1, 3, 8]),
                }
                for idx, value in enumerate(key_values[:8])
            ],
        )
    )
    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    sum_alias = make_safe_output_name("sum_lookup_j", used=used_columns)
    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": tag_col, "cmp": "in_set", "value": ["dim-a", "space value"]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {"op": "fill_null", "column": value_col, "value": 0},
            {
                "op": "groupby",
                "keys": [bucket_col],
                "aggs": [
                    {"column": key, "func": "count", "as": count_alias},
                    {"column": value_col, "func": "sum", "as": sum_alias},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_case_membership:{key}:bucket={bucket_col}"


def _append_empty_filter_global_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_empty_filter_global_aggregate:none"
    available = _available_columns(tables, operations)
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not numeric:
        return "append_empty_filter_global_aggregate:no-numeric-column"
    source = rnd.choice(numeric)
    source_type = _column_type(tables, source)
    used_columns = set(available)
    operations.append(
        {
            "op": "filter",
            "column": source,
            "cmp": ">",
            "value": 10**12 if source_type == "int" else 1.0e12,
        }
    )
    aggs: list[dict[str, Any]] = []
    count_alias = make_safe_output_name(f"count_{source}_empty", used=used_columns)
    used_columns.add(count_alias)
    aggs.append({"column": source, "func": "count", "as": count_alias})
    if numeric:
        numeric_source = source if source in numeric else rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}_empty", used=used_columns)
        used_columns.add(sum_alias)
        mean_alias = make_safe_output_name(f"mean_{numeric_source}_empty", used=used_columns)
        used_columns.add(mean_alias)
        aggs.extend(
            [
                {"column": numeric_source, "func": "sum", "as": sum_alias},
                {"column": numeric_source, "func": "mean", "as": mean_alias},
            ]
        )
    if bools:
        bool_source = source if source in bools else rnd.choice(bools)
        any_alias = make_safe_output_name(f"any_{bool_source}_empty", used=used_columns)
        used_columns.add(any_alias)
        all_alias = make_safe_output_name(f"all_{bool_source}_empty", used=used_columns)
        aggs.extend(
            [
                {"column": bool_source, "func": "any", "as": any_alias},
                {"column": bool_source, "func": "all", "as": all_alias},
            ]
        )
    operations.append({"op": "aggregate", "aggs": aggs})
    return f"append_empty_filter_global_aggregate:{source}:{','.join(agg['as'] for agg in aggs)}"


def _append_sql_union_coalesce_distinct_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_sql_union_coalesce_distinct_topk:none"
    available = _available_columns(tables, operations)
    strings = [column for column in available if _column_type(tables, column) == "str"]
    if not strings:
        return "append_sql_union_coalesce_distinct_topk:no-string-column"
    table_name = make_safe_output_name("t_union_distinct_mut", used={table.name for table in tables})
    columns = [ColumnSpec(column, _column_type(tables, column), nullable=True) for column in available]
    row_count = rnd.randint(2, 5)
    rows = []
    for row_idx in range(row_count):
        row: dict[str, Any] = {}
        for column in columns:
            row[column.name] = None if row_idx % 4 == 0 else _literal_for_type(column.type, rnd)
        rows.append(row)
    tables.append(TableData(table_name, columns, rows))

    used_columns = set(available)
    source = rnd.choice(strings)
    nonempty_col = make_safe_output_name(f"{source}_nonempty", used=used_columns)
    used_columns.add(nonempty_col)
    label_col = nonempty_col
    operations.extend(
        [
            {"op": "union_all", "table": table_name},
            {"op": "mutate", "column": nonempty_col, "expr": {"kind": "string_null_if_empty", "source": source}},
        ]
    )
    coalesce_sources = [column for column in strings if column != source]
    if coalesce_sources:
        label_col = make_safe_output_name(f"{source}_label", used=used_columns)
        used_columns.add(label_col)
        operations.append(
            {
                "op": "coalesce",
                "columns": [nonempty_col, rnd.choice(coalesce_sources)],
                "as": label_col,
                "fallback": "missing",
            }
        )
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    distinct_cols = unique_preserve_order([label_col, *(bools + numeric)[:2]])
    operations.extend(
        [
            {"op": "distinct", "columns": distinct_cols},
            {
                "op": "sort",
                "keys": [
                    {"column": distinct_cols[0], "ascending": True, "nulls": "first"},
                    *[
                        {
                            "column": column,
                            "ascending": False,
                            "nulls": "first" if idx == 0 else "last",
                        }
                        for idx, column in enumerate(distinct_cols[1:])
                    ],
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 2)},
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + row_count + 2, 8)))},
        ]
    )
    return f"append_sql_union_coalesce_distinct_topk:{table_name}:{source}:label={label_col}"


def _append_coalesce_sort_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_coalesce_sort_topk:none"
    context = _mutation_operation_context(tables, operations)
    if context is None:
        return "append_coalesce_sort_topk:no-context"
    groups = _coalesce_column_groups(context)
    if not groups:
        return "append_coalesce_sort_topk:no-compatible-columns"
    output_type, candidates = rnd.choice(groups)
    width = rnd.randint(2, min(3, len(candidates)))
    columns = rnd.sample(candidates, width)
    used_columns = set(context.available) | {op.get("as", "") for op in operations if isinstance(op, dict)}
    alias = make_safe_output_name(f"co_topk_{columns[0]}", used=used_columns)
    used_columns.add(alias)
    coalesce_op: dict[str, Any] = {"op": "coalesce", "columns": columns, "as": alias}
    if rnd.random() < 0.85:
        coalesce_op["fallback"] = _literal_for_type(output_type, rnd)
    sort_keys = [
        {
            "column": alias,
            "ascending": rnd.choice([True, False]),
            "nulls": rnd.choice(["first", "last"]),
        }
    ]
    tie_candidates = [column for column in context.available if column not in columns]
    if tie_candidates and rnd.random() < 0.80:
        tie_column = rnd.choice(tie_candidates)
        sort_keys.append(
            {
                "column": tie_column,
                "ascending": rnd.choice([True, False]),
                "nulls": rnd.choice(["first", "last"]),
            }
        )
    operations.extend(
        [
            coalesce_op,
            {"op": "sort", "keys": sort_keys},
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_coalesce_sort_topk:{','.join(columns)}:as={alias}"


def _append_boolean_membership_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_membership_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_boolean_membership_case_aggregate:no-bool-column"
    source = "flag" if "flag" in bools else rnd.choice(bools)
    present_values = unique_preserve_order(
        bool(row.get(source))
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), bool)
    )
    if not present_values:
        present_values = [True]
    selected_value = rnd.choice(present_values)
    table_name = make_safe_output_name("t_bool_membership_mut", used={table.name for table in tables})
    right_key = "flag_key"
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(right_key, "bool", nullable=False)],
            [{right_key: selected_value}],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{source}_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": source, "func": "count", "as": count_alias},
        {"column": source, "func": "any", "as": any_alias},
        {"column": source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "filter", "column": source, "cmp": "bool_is_not_unknown", "value": None},
            {"op": "semi_join", "table": table_name, "left_on": source, "right_on": right_key},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": source, "cmp": "bool_is_true", "value": None},
                "then": "true_member",
                "else": "false_member",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_boolean_membership_case_aggregate:{source}:member={selected_value}:bucket={bucket_col}"


def _append_left_join_boolean_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_case_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_case_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_case_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    tag_col = make_safe_output_name("lookup_tag", used=used_columns)
    used_columns.add(tag_col)
    missing_col = make_safe_output_name("dim_missing", used=used_columns)
    used_columns.add(missing_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(tag_col, "str", nullable=True),
            ],
            [
                {
                    right_key: value,
                    tag_col: None if idx % 3 == 0 else rnd.choice(["matched", "space value"]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": bool_source, "func": "any", "as": any_alias},
        {"column": bool_source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {
                "op": "case_when",
                "as": missing_col,
                "condition": {"column": tag_col, "cmp": "is_null", "value": None},
                "then": True,
                "else": False,
            },
            {"op": "groupby", "keys": [missing_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": missing_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_boolean_case_aggregate:{key}:bool={bool_source}:missing={missing_col}"


def _append_left_join_boolean_coalesce_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_coalesce_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_coalesce_case_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    fallback_bool = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_coalesce_case_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_coalesce_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    dim_flag_col = make_safe_output_name("dim_flag", used=used_columns)
    used_columns.add(dim_flag_col)
    effective_col = make_safe_output_name("flag_effective", used=used_columns)
    used_columns.add(effective_col)
    bucket_col = make_safe_output_name("flag_bucket", used=used_columns)
    used_columns.add(bucket_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(dim_flag_col, "bool", nullable=True),
            ],
            [
                {
                    right_key: value,
                    dim_flag_col: None if idx % 3 == 0 else rnd.choice([True, False]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{effective_col}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{effective_col}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": effective_col, "func": "any", "as": any_alias},
        {"column": effective_col, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [dim_flag_col, fallback_bool], "as": effective_col, "fallback": False},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": effective_col, "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false_or_missing",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_boolean_coalesce_case_aggregate:{key}:fallback={fallback_bool}:effective={effective_col}"


def _append_boolean_antijoin_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_antijoin_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_boolean_antijoin_case_aggregate:no-bool-column"
    source = "flag" if "flag" in bools else rnd.choice(bools)
    present_values = unique_preserve_order(
        bool(row.get(source))
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), bool)
    )
    if not present_values:
        present_values = [True]
    excluded_value = rnd.choice(present_values)
    table_name = make_safe_output_name("t_bool_antijoin_mut", used={table.name for table in tables})
    right_key = "flag_key"
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(right_key, "bool", nullable=False)],
            [{right_key: excluded_value}],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{source}_anti_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": source, "func": "count", "as": count_alias},
        {"column": source, "func": "any", "as": any_alias},
        {"column": source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "filter", "column": source, "cmp": "bool_is_not_unknown", "value": None},
            {"op": "anti_join", "table": table_name, "left_on": source, "right_on": right_key},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": source, "cmp": "bool_is_true", "value": None},
                "then": "true_non_member",
                "else": "false_non_member",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_boolean_antijoin_case_aggregate:{source}:excluded={excluded_value}:bucket={bucket_col}"


def _append_left_join_boolean_coalesce_filter_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_coalesce_filter_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_coalesce_filter_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    fallback_bool = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_coalesce_filter_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_filter_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    dim_flag_col = make_safe_output_name("dim_flag", used=used_columns)
    used_columns.add(dim_flag_col)
    effective_col = make_safe_output_name("flag_effective", used=used_columns)
    used_columns.add(effective_col)
    bucket_col = make_safe_output_name("flag_filter_bucket", used=used_columns)
    used_columns.add(bucket_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(dim_flag_col, "bool", nullable=True),
            ],
            [
                {
                    right_key: value,
                    dim_flag_col: None if idx % 4 == 0 else rnd.choice([True, False]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    filter_value = rnd.choice([True, False])
    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{effective_col}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{effective_col}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": effective_col, "func": "any", "as": any_alias},
        {"column": effective_col, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [dim_flag_col, fallback_bool], "as": effective_col, "fallback": False},
            {"op": "filter", "column": effective_col, "cmp": "==", "value": filter_value},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": effective_col, "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return (
        "append_left_join_boolean_coalesce_filter_aggregate:"
        f"{key}:fallback={fallback_bool}:filter={filter_value}:effective={effective_col}"
    )


def _append_numeric_text_boolean_antijoin_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_numeric_text_boolean_antijoin_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_numeric_text_boolean_antijoin_case_aggregate:no-bool-column"
    used_columns = set(available)
    numeric_text = _ensure_numeric_string_column(tables, available, rnd, used_columns)
    if numeric_text is None:
        return "append_numeric_text_boolean_antijoin_case_aggregate:no-numeric-string-column"
    if numeric_text not in available:
        available.append(numeric_text)
    used_columns.add(numeric_text)

    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    num_col = make_safe_output_name("num_value", used=used_columns)
    used_columns.add(num_col)
    table_name = make_safe_output_name("t_numeric_antijoin_mut", used={table.name for table in tables})
    parsed_values = []
    for row in tables[0].rows:
        value = row.get(numeric_text)
        if value is None:
            continue
        try:
            parsed_values.append(int(value))
        except (TypeError, ValueError):
            continue
    parsed_values = unique_preserve_order(parsed_values)
    if not parsed_values:
        parsed_values = [0]
    excluded_values = parsed_values[: max(1, min(len(parsed_values), 3))]
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(num_col, "int", nullable=False)],
            [{num_col: value} for value in excluded_values],
        )
    )

    bucket_col = make_safe_output_name(f"{bool_source}_num_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    sum_alias = make_safe_output_name(f"sum_{num_col}", used=used_columns)
    used_columns.add(sum_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)

    operations.extend(
        [
            {
                "op": "mutate",
                "column": num_col,
                "expr": {"kind": "cast", "source": numeric_text, "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": num_col, "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": bool_source, "cmp": "bool_is_not_false", "value": None},
            {"op": "anti_join", "table": table_name, "left_on": num_col, "right_on": num_col},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": bool_source, "cmp": "bool_is_true", "value": None},
                "then": "true_unmatched_number",
                "else": "null_or_false_unmatched_number",
            },
            {
                "op": "groupby",
                "keys": [bucket_col],
                "aggs": [
                    {"column": bool_source, "func": "count", "as": count_alias},
                    {"column": num_col, "func": "sum", "as": sum_alias},
                    {"column": bool_source, "func": "any", "as": any_alias},
                    {"column": bool_source, "func": "all", "as": all_alias},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return (
        "append_numeric_text_boolean_antijoin_case_aggregate:"
        f"{numeric_text}:bool={bool_source}:excluded={excluded_values}"
    )


def _append_multi_key_membership_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_multi_key_membership_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    int_keys = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "int"
    ]
    string_keys = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "str"
    ]
    bools = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "bool"
    ]
    if not int_keys or not string_keys or not bools:
        return "append_multi_key_membership_case_aggregate:no-key-or-bool-column"

    left_keys = ["id" if "id" in int_keys else rnd.choice(int_keys), "g" if "g" in string_keys else rnd.choice(string_keys)]
    if left_keys[0] == left_keys[1]:
        return "append_multi_key_membership_case_aggregate:duplicate-keys"
    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(
        tuple(row.get(column) for column in left_keys)
        for row in tables[0].rows
        if all(row.get(column) is not None for column in left_keys)
    )
    if len(key_values) < 2:
        return "append_multi_key_membership_case_aggregate:not-enough-key-values"

    table_name = make_safe_output_name("t_multi_key_membership_mut", used={table.name for table in tables})
    selected_values = key_values[: max(1, len(key_values) // 2)]
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(column, base_columns[column].type, nullable=False) for column in left_keys],
            [dict(zip(left_keys, values)) for values in selected_values],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{bool_source}_multi_key_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": left_keys[0], "func": "count", "as": count_alias},
        {"column": bool_source, "func": "any", "as": any_alias},
        {"column": bool_source, "func": "all", "as": all_alias},
    ]
    numeric = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "float"} and column not in left_keys
    ]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    join_kind = rnd.choice(["semi_join", "anti_join"])
    operations.extend(
        [
            {"op": "filter", "column": left_keys[1], "cmp": "is_not_null", "value": None},
            {"op": join_kind, "table": table_name, "left_on": left_keys, "right_on": left_keys},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": bool_source, "cmp": "bool_is_true", "value": None},
                "then": f"{join_kind}_true",
                "else": f"{join_kind}_false_or_null",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_multi_key_membership_case_aggregate:{join_kind}:{','.join(left_keys)}"


def _append_join_filter_groupby_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    context = _mutation_operation_context(tables, operations)
    if context is None:
        return "append_join_filter_groupby_topk:none"
    membership_pairs = _compatible_membership_key_pairs_for_context(context)
    if not membership_pairs:
        return "append_join_filter_groupby_topk:no-compatible-right-table"
    right, left_columns, right_columns = _choose_membership_key_pair(membership_pairs, rnd)
    group_candidates = [
        column
        for column in context.available
        if context.column_type(column) in {"int", "str", "bool"}
    ]
    group_keys = unique_preserve_order(
        [
            *left_columns,
            *(
                [rnd.choice([column for column in group_candidates if column not in left_columns])]
                if len(left_columns) < 2 and any(column not in left_columns for column in group_candidates)
                else []
            ),
        ]
    )
    if not group_keys:
        return "append_join_filter_groupby_topk:no-group-key"

    value_candidates = [column for column in [*context.numeric, *context.bools] if column not in group_keys]
    if not value_candidates:
        value_candidates = [*context.numeric, *context.bools]
    if not value_candidates:
        return "append_join_filter_groupby_topk:no-aggregate-column"

    filter_candidates = [
        column
        for column in [*context.numeric, *context.bools, *group_keys]
        if column in context.available
    ]
    filter_column = rnd.choice(filter_candidates or group_keys)
    filter_type = context.column_type(filter_column)
    if filter_type == "bool":
        filter_op = {"op": "filter", "column": filter_column, "cmp": "bool_is_not_false", "value": None}
    elif filter_type in {"int", "float"}:
        filter_cmp = rnd.choice([">=", "!=", "range_closed"])
        filter_op = {
            "op": "filter",
            "column": filter_column,
            "cmp": filter_cmp,
            "value": (
                sorted(rnd.sample(_literal_list_for_type(filter_type, rnd), 2))
                if filter_cmp == "range_closed"
                else _literal_for_type(filter_type, rnd)
            ),
        }
    else:
        filter_op = {"op": "filter", "column": filter_column, "cmp": "is_not_null", "value": None}

    used_columns = set(context.available) | set(group_keys)
    count_alias = make_safe_output_name("count_joined_rows", used=used_columns)
    used_columns.add(count_alias)
    value_column = rnd.choice(value_candidates)
    if context.column_type(value_column) == "bool":
        value_func = rnd.choice(["any", "all", "count", "nunique"])
    else:
        value_func = rnd.choice(["sum", "min", "max", "count", "nunique"])
    value_alias = make_safe_output_name(f"{value_func}_{value_column}", used=used_columns)
    aggs = [
        {"column": group_keys[0], "func": "count", "as": count_alias},
        {"column": value_column, "func": value_func, "as": value_alias},
    ]
    projection = unique_preserve_order([*group_keys, count_alias, value_alias])
    operations.extend(
        [
            {
                "op": "join",
                "table": right.name,
                "left_on": join_key_arg(left_columns),
                "right_on": join_key_arg(right_columns),
                "how": rnd.choice(["inner", "left"]),
            },
            filter_op,
            {"op": "groupby", "keys": group_keys, "aggs": aggs},
            {"op": "select", "columns": projection},
            {
                "op": "sort",
                "keys": [
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                    {"column": group_keys[0], "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(1, 4)},
        ]
    )
    return f"append_join_filter_groupby_topk:{right.name}:{','.join(group_keys)}"


def _case_when_literals_for_type(typ: str, rnd: random.Random) -> tuple[Any, Any]:
    if typ == "int":
        return rnd.choice([1, 10]), rnd.choice([0, -1])
    if typ == "float":
        return rnd.choice([1.0, 0.5]), rnd.choice([0.0, -1.0])
    if typ == "bool":
        return True, False
    return rnd.choice(["matched", "high", "yes"]), rnd.choice(["other", "low", "no"])


def _coalesce_column_groups(context: MutationOperationContext) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for column in context.available:
        typ = context.column_type(column)
        if typ in {"int", "float", "str", "bool"}:
            groups.setdefault(typ, []).append(column)
    return [(typ, columns) for typ, columns in groups.items() if len(columns) >= 2]


def _compatible_membership_key_pairs_for_context(
    context: MutationOperationContext,
) -> list[tuple[TableData, list[str], list[str]]]:
    pairs: list[tuple[TableData, list[str], list[str]]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    available = list(context.available)
    for extra in context.extra_tables:
        right_types = {column.name: column.type for column in extra.columns}
        matching = [
            column
            for column in available
            if right_types.get(column) is not None and right_types.get(column) == context.column_type(column)
        ]
        for width in range(1, min(2, len(matching)) + 1):
            for key_columns in combinations(matching, width):
                left_columns = list(key_columns)
                right_columns = list(key_columns)
                key = (extra.name, tuple(left_columns), tuple(right_columns))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((extra, left_columns, right_columns))
    return pairs


def _choose_membership_key_pair(
    pairs: list[tuple[TableData, list[str], list[str]]],
    rnd: random.Random,
    *,
    prefer_multi_key: bool = True,
) -> tuple[TableData, list[str], list[str]]:
    if prefer_multi_key:
        multi_key_pairs = [pair for pair in pairs if len(pair[1]) > 1]
        if multi_key_pairs and rnd.random() < 0.65:
            return rnd.choice(multi_key_pairs)
    return rnd.choice(pairs)


def _mutation_operation_context(
    tables: list[TableData],
    operations: list[dict[str, Any]],
) -> MutationOperationContext | None:
    cache = _ACTIVE_MUTATION_STATE_CACHE.get()
    cache_key = _mutation_state_cache_key(tables, operations, cache=cache) if cache is not None else None
    if cache is not None and cache_key is not None and cache_key in cache.context_by_key:
        return cache.context_by_key[cache_key]
    if not tables:
        if cache is not None and cache_key is not None:
            cache.context_by_key[cache_key] = None
        return None
    schema = _mutation_schema(tables, operations, cache_key=cache_key)
    available = tuple(schema.available)
    if not available:
        if cache is not None and cache_key is not None:
            cache.context_by_key[cache_key] = None
        return None
    numeric = tuple(
        column
        for column in available
        if schema.column_type(column) in {"int", "float"} or column.startswith(NUMERIC_ALIAS_PREFIXES)
    )
    bools = tuple(
        column
        for column in available
        if schema.column_type(column) == "bool" or column.startswith(BOOLEAN_ALIAS_PREFIXES)
    )
    strings = tuple(column for column in available if schema.column_type(column) == "str")
    context = MutationOperationContext(
        table=tables[0],
        schema=schema,
        available=available,
        numeric=numeric,
        bools=bools,
        strings=strings,
        date_strings=tuple(column for column in strings if _looks_like_date_column(column)),
        numeric_strings=tuple(column for column in strings if _looks_like_numeric_string_column(column)),
        extra_tables=tuple(tables[1:]),
    )
    if cache is not None and cache_key is not None:
        cache.context_by_key[cache_key] = context
    return context


def _mutation_state_cache_key(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    *,
    cache: _MutationStateCache | None = None,
) -> tuple[Any, ...]:
    return (
        tuple(_mutation_table_cache_key_cached(table, cache=cache) for table in tables),
        tuple(_mutation_operation_cache_key_cached(operation, cache=cache) for operation in operations),
    )


def _mutation_table_cache_key_cached(
    table: TableData,
    *,
    cache: _MutationStateCache | None = None,
) -> tuple[Any, ...]:
    if cache is None:
        return _mutation_table_cache_key(table)
    table_id = id(table)
    cached = cache.table_key_by_identity.get(table_id)
    if cached is not None:
        return cached
    key = _mutation_table_cache_key(table)
    cache.table_key_by_identity[table_id] = key
    return key


def _invalidate_table_cache(cache: _MutationStateCache, tables: list[TableData]) -> None:
    for table in tables:
        cache.table_key_by_identity.pop(id(table), None)


def _mutation_operation_cache_key_cached(
    operation: Any,
    *,
    cache: _MutationStateCache | None = None,
) -> Any:
    if cache is None:
        return _mutation_operation_cache_key(operation)
    operation_id = id(operation)
    cached = cache.operation_key_by_identity.get(operation_id)
    if cached is not None:
        cached_operation, cached_key = cached
        if cached_operation is operation:
            return cached_key
    key = _mutation_operation_cache_key(operation)
    cache.operation_key_by_identity[operation_id] = (operation, key)
    return key


def _invalidate_operation_cache(cache: _MutationStateCache, operations: Sequence[Any]) -> None:
    for operation in operations:
        operation_id = id(operation)
        cached = cache.operation_key_by_identity.get(operation_id)
        if cached is not None and cached[0] is operation:
            cache.operation_key_by_identity.pop(operation_id, None)


def _mutation_table_cache_key(table: TableData) -> tuple[Any, ...]:
    column_specs = tuple((column.name, column.type, bool(column.nullable)) for column in table.columns)
    null_columns = _table_null_columns(table)
    null_flags = tuple(
        (
            column.name,
            column.name in null_columns,
        )
        for column in table.columns
    )
    return (table.name, column_specs, null_flags)


def _table_null_columns(table: TableData) -> set[str]:
    column_names = {column.name for column in table.columns}
    null_columns: set[str] = set()
    for row in table.rows:
        for key, value in row.items():
            if value is None and key in column_names:
                null_columns.add(key)
        if len(null_columns) >= len(column_names):
            break
    return null_columns


def _mutation_operation_cache_key(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf", 1 if value > 0 else -1)
        return value
    if isinstance(value, IRNode):
        value = value.to_dict()
    if type(value) is dict:
        return tuple([(str(key), _mutation_operation_cache_key(item)) for key, item in value.items()])
    if type(value) is list:
        return tuple([_mutation_operation_cache_key(item) for item in value])
    if type(value) is tuple:
        return tuple([_mutation_operation_cache_key(item) for item in value])
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _mutation_operation_cache_key(item))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return tuple(_mutation_operation_cache_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_mutation_operation_cache_key(item) for item in value)
    if isinstance(value, set):
        normalized = [_mutation_operation_cache_key(item) for item in value]
        return tuple(sorted(normalized, key=repr))
    return value


def _random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> dict[str, Any] | None:
    context = _mutation_operation_context(tables, operations)
    if context is None:
        return None
    table = context.table
    available = context.available
    numeric = context.numeric
    bools = context.bools
    strings = context.strings
    date_strings = context.date_strings
    numeric_strings = context.numeric_strings
    membership_pairs = _compatible_membership_key_pairs_for_context(context)
    base_columns = {column.name for column in table.columns}
    nullable_columns = {
        column
        for column in available
        if context.column_has_null(column) or column not in base_columns
    }
    comparable = tuple(
        column
        for column in available
        if context.column_type(column) in {"int", "float", "str", "bool"}
    )
    derived_comparable = tuple(column for column in comparable if column not in base_columns)
    coalesce_groups = _coalesce_column_groups(context)
    choices = ["filter", "select", "sort", "limit", "offset", "distinct"]
    if available:
        choices.append("fill_null")
    if comparable:
        choices.extend(["case_when", "row_number_filter"])
    if membership_pairs:
        choices.extend(["semi_join", "anti_join", "tuple_absence_filter"])
    if coalesce_groups:
        choices.append("coalesce")
    if numeric or bools or strings:
        choices.append("mutate")
    if numeric:
        choices.append("running_sum")
    if numeric or bools:
        choices.append("groupby")
        choices.append("aggregate")
    kind = rnd.choice(choices)
    if kind == "filter":
        if derived_comparable and rnd.random() < 0.55:
            col = rnd.choice(derived_comparable)
        else:
            nullable_candidates = [column for column in comparable if column in nullable_columns]
            col = rnd.choice(nullable_candidates) if nullable_candidates and rnd.random() < 0.30 else rnd.choice(available)
        typ = context.column_type(col)
        cmp = rnd.choice(["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="])
        if typ in {"int", "float"} and rnd.random() < 0.20:
            cmp = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
        if rnd.random() < 0.15:
            cmp = rnd.choice(["in_set", "not_in_set"])
        if col in nullable_columns and rnd.random() < 0.25:
            cmp = rnd.choice(["is_null", "is_not_null"])
        if typ == "bool" and rnd.random() < 0.25:
            cmp = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
        if typ in {"int", "float"} and rnd.random() < 0.18:
            cmp = "range_closed"
        if cmp in {"in_set", "not_in_set"}:
            value = _literal_list_for_type(typ, rnd)
        elif cmp == "range_closed":
            value = sorted(rnd.sample(_literal_list_for_type(typ, rnd), 2))
        elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
            value = None
        else:
            value = _literal_for_type(typ, rnd)
        return {"op": "filter", "column": col, "cmp": cmp, "value": value}
    if kind == "select":
        count = rnd.randint(1, len(available))
        return {"op": "select", "columns": sorted(rnd.sample(available, count))}
    if kind == "sort":
        first = rnd.choice(available)
        cols = [first] + sorted(c for c in available if c != first)
        if len(cols) > 1 and rnd.random() < 0.35:
            return {
                "op": "sort",
                "keys": [
                    {
                        "column": column,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                    for column in cols
                ],
            }
        return {"op": "sort", "columns": cols, "ascending": rnd.choice([True, False])}
    if kind == "limit":
        return {"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 3))}
    if kind == "offset":
        return {"op": "offset", "n": rnd.randint(0, max(1, len(table.rows) + 3))}
    if kind == "distinct":
        count = rnd.randint(1, min(3, len(available)))
        return {"op": "distinct", "columns": sorted(rnd.sample(available, count))}
    if kind in {"semi_join", "anti_join"} and membership_pairs:
        right, left_columns, right_columns = _choose_membership_key_pair(membership_pairs, rnd)
        return {
            "op": kind,
            "table": right.name,
            "left_on": join_key_arg(left_columns),
            "right_on": join_key_arg(right_columns),
        }
    if kind == "tuple_absence_filter" and membership_pairs:
        right, left_columns, right_columns = _choose_membership_key_pair(membership_pairs, rnd)
        return {
            "op": "tuple_absence_filter",
            "columns": left_columns,
            "table": right.name,
            "right_columns": right_columns,
        }
    if kind == "fill_null":
        column = rnd.choice(available)
        return {"op": "fill_null", "column": column, "value": _literal_for_type(context.column_type(column), rnd)}
    if kind == "coalesce" and coalesce_groups:
        output_type, candidates = rnd.choice(coalesce_groups)
        width = rnd.randint(2, min(3, len(candidates)))
        columns = rnd.sample(candidates, width)
        alias = make_safe_output_name(
            f"co_{columns[0]}",
            used=set(available) | {op.get("as", "") for op in operations if isinstance(op, dict)},
        )
        op: dict[str, Any] = {"op": "coalesce", "columns": columns, "as": alias}
        if rnd.random() < 0.75:
            op["fallback"] = _literal_for_type(output_type, rnd)
        return op
    if kind == "case_when" and comparable:
        if derived_comparable and rnd.random() < 0.60:
            predicate_col = rnd.choice(derived_comparable)
        else:
            nullable_candidates = [column for column in comparable if column in nullable_columns]
            predicate_col = (
                rnd.choice(nullable_candidates)
                if nullable_candidates and rnd.random() < 0.30
                else rnd.choice(comparable)
            )
        predicate_type = context.column_type(predicate_col)
        cmp_ops = ["==", "!="] if predicate_type in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
        if predicate_type in {"int", "float"} and rnd.random() < 0.25:
            cmp_ops = [rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"]), *cmp_ops]
        if predicate_type == "bool" and rnd.random() < 0.35:
            cmp_ops = [rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]), *cmp_ops]
        if predicate_type in {"int", "float"} and rnd.random() < 0.20:
            cmp_ops = ["range_closed", *cmp_ops]
        if predicate_col in nullable_columns and rnd.random() < 0.25:
            cmp_ops = [rnd.choice(["is_null", "is_not_null"]), *cmp_ops]
        cmp = rnd.choice(cmp_ops)
        if cmp == "range_closed":
            predicate_value = sorted(rnd.sample(_literal_list_for_type(predicate_type, rnd), 2))
        elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
            predicate_value = None
        else:
            predicate_value = _literal_for_type(predicate_type, rnd)
        output_type = rnd.choice(["str", "int", "bool"])
        then_value, else_value = _case_when_literals_for_type(output_type, rnd)
        alias = make_safe_output_name(f"cw_{operation_count(operations, 'case_when')}", used=set(available))
        return {
            "op": "case_when",
            "as": alias,
            "condition": {"column": predicate_col, "cmp": cmp, "value": predicate_value},
            "then": then_value,
            "else": else_value,
        }
    if kind == "row_number_filter" and comparable:
        order_width = 1 if len(comparable) == 1 else rnd.randint(2, min(3, len(comparable)))
        if derived_comparable and rnd.random() < 0.55:
            first_order = rnd.choice(derived_comparable)
            remaining_order = [column for column in comparable if column != first_order]
            order_by_columns = [first_order, *rnd.sample(remaining_order, k=max(0, order_width - 1))]
        else:
            order_by_columns = rnd.sample(list(comparable), k=order_width)
        partition_candidates = [column for column in available if context.column_type(column) in {"int", "str", "bool"}]
        partition_by: list[str] = []
        if partition_candidates and rnd.random() < 0.8:
            partition_width = rnd.randint(1, min(2, len(partition_candidates)))
            partition_by = unique_preserve_order(rnd.sample(partition_candidates, k=partition_width))
        comparator = "==" if rnd.random() < 0.45 else rnd.choice(["<", "<="])
        value = 1 if comparator == "==" else rnd.choice([1, 2, 3])
        return {
            "op": "row_number_filter",
            "partition_by": partition_by,
            "order_by": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": (
                        "first"
                        if column in nullable_columns and rnd.random() < 0.60
                        else rnd.choice(["first", "last"])
                    ),
                }
                for column in order_by_columns
            ],
            "cmp": comparator,
            "value": value,
        }
    if kind == "running_sum" and numeric:
        derived_numeric = [column for column in numeric if column not in base_columns]
        if derived_numeric and rnd.random() < 0.45:
            source = rnd.choice(derived_numeric)
        else:
            source = rnd.choice(numeric)
        order_width = 1 if len(comparable) == 1 else rnd.randint(2, min(3, len(comparable)))
        if derived_comparable and rnd.random() < 0.55:
            first_order = rnd.choice(derived_comparable)
            remaining_order = [column for column in comparable if column != first_order]
            order_by_columns = [first_order, *rnd.sample(remaining_order, k=max(0, order_width - 1))]
        else:
            order_by_columns = rnd.sample(list(comparable), k=order_width)
        partition_candidates = [
            column
            for column in available
            if column != source and context.column_type(column) in {"int", "str", "bool"}
        ]
        partition_by: list[str] = []
        if partition_candidates and rnd.random() < 0.75:
            partition_width = rnd.randint(1, min(2, len(partition_candidates)))
            partition_by = unique_preserve_order(rnd.sample(partition_candidates, k=partition_width))
        op: dict[str, Any] = {
            "op": "running_sum",
            "source": source,
            "column": make_safe_output_name(f"run_{source}", used=set(available)),
            "order_by": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": (
                        "first"
                        if column in nullable_columns and rnd.random() < 0.60
                        else rnd.choice(["first", "last"])
                    ),
                }
                for column in order_by_columns
            ],
            "input_dtype": "float32" if context.column_type(source) == "float" or rnd.random() < 0.5 else "float64",
        }
        if partition_by:
            op["partition_by"] = partition_by
        return op
    if kind == "mutate" and (numeric or bools or strings):
        if numeric_strings and rnd.random() < 0.20:
            src = rnd.choice(numeric_strings)
            expr = {
                "kind": "cast",
                "source": src,
                "to": rnd.choice(["int", "float"]),
                "input_domain": "integer_string",
            }
        elif date_strings and rnd.random() < 0.25:
            src = rnd.choice(date_strings)
            expr = {"kind": "date_part", "source": src, "part": rnd.choice(["year", "month", "day"])}
        elif bools and (not numeric and not strings or rnd.random() < 0.20):
            expr = {"kind": "bool_not", "source": rnd.choice(bools)}
        elif strings and (not numeric or rnd.random() < 0.3):
            src = rnd.choice(strings)
            string_exprs = [
                {"kind": "string_length", "source": src},
                {"kind": "string_lower", "source": src},
                {"kind": "string_upper", "source": src},
                {"kind": "string_strip", "source": src},
                {"kind": "string_null_if_empty", "source": src},
                {"kind": "string_replace", "source": src, "old": " ", "new": "_"},
                {"kind": "string_slice", "source": src, "start": 0, "length": rnd.randint(1, 3)},
                {"kind": "string_split_part", "source": src, "sep": rnd.choice([" ", "-", "_"]), "index": 0},
                {"kind": "string_basename", "source": src},
                {"kind": "string_contains", "source": src, "needle": rnd.choice(["a", "A", "space", "pad"])},
                {"kind": "string_starts_with", "source": src, "needle": rnd.choice(["a", "A", "space", "pad"])},
                {"kind": "string_ends_with", "source": src, "needle": rnd.choice(["a", "A", "e", "d"])},
            ]
            if len(strings) > 1:
                other = rnd.choice([column for column in strings if column != src])
                string_exprs.append({"kind": "string_concat", "source": src, "other": other, "sep": "-"})
            expr = rnd.choice(string_exprs)
        else:
            src = rnd.choice(numeric)
            if len(numeric) > 1 and rnd.random() < 0.20:
                numerator = rnd.choice([column for column in numeric if column != src])
                expr = {"kind": "reverse_division_columns", "source": src, "numerator": numerator}
            elif rnd.random() < 0.20:
                expr = {"kind": "abs", "source": src}
            elif rnd.random() < 0.20:
                if context.column_type(src) == "int":
                    lower, upper = sorted(rnd.sample([-10, -5, -2, 0, 2, 5, 10], 2))
                else:
                    lower, upper = sorted(rnd.sample([-10.0, -2.5, -1.0, 0.0, 1.0, 2.5, 10.0], 2))
                expr = {"kind": "clip", "source": src, "lower": lower, "upper": upper}
            elif rnd.random() < 0.25:
                if context.column_type(src) == "int":
                    expr = {"kind": "cast", "source": src, "to": rnd.choice(["float", "str"])}
                else:
                    expr = {"kind": "cast", "source": src, "to": "float"}
            else:
                expr = {"kind": "add_const", "source": src, "value": rnd.choice([-10, -1, 0, 1, 10])}
        return {
            "op": "mutate",
            "column": f"m_{operation_count(operations, 'mutate')}",
            "expr": expr,
        }
    if kind == "groupby" and (numeric or bools):
        key_candidates = [column for column in available if context.column_type(column) in {"int", "str", "bool"}]
        keys = [rnd.choice(available)]
        if key_candidates:
            key_count = 1 if len(key_candidates) == 1 or rnd.random() < 0.8 else 2
            keys = sorted(rnd.sample(key_candidates, key_count))
        val = rnd.choice(numeric + bools)
        if val in bools:
            func = rnd.choice(["any", "all", "min", "max", "count", "nunique"])
        else:
            func = rnd.choice(["sum", "mean", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}", used=set(keys))
        return {"op": "groupby", "keys": keys, "aggs": [{"column": val, "func": func, "as": alias}]}
    if kind == "aggregate" and (numeric or bools):
        val = rnd.choice(numeric + bools)
        if val in bools:
            func = rnd.choice(["any", "all", "min", "max", "count", "nunique"])
        else:
            func = rnd.choice(["sum", "mean", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}_all")
        return {"op": "aggregate", "aggs": [{"column": val, "func": func, "as": alias}]}
    return None


def _tweak_operation(tables: list[TableData], op: dict[str, Any], rnd: random.Random) -> str:
    kind = op_kind(op)
    if kind == "filter":
        op["cmp"] = rnd.choice([">", ">=", "<", "<=", "==", "!="])
        op["value"] = _literal_for_type(_column_type(tables, op["column"]), rnd)
        return "tweak:filter"
    elif kind == "sort":
        if "keys" in op:
            keys = [key.to_dict() for key in normalize_sort_keys(op)]
            if not keys:
                return "tweak:sort:no-keys"
            key = keys[rnd.randrange(len(keys))]
            if rnd.random() < 0.5:
                key["ascending"] = not bool(key.get("ascending", True))
            else:
                key["nulls"] = "first" if key.get("nulls", "last") == "last" else "last"
            op["keys"] = keys
        else:
            op["ascending"] = not op_ascending(op, True)
        return "tweak:sort"
    elif kind == "limit":
        op["n"] = max(0, op_n(op, 0) + rnd.choice([-2, -1, 1, 2]))
        return "tweak:limit"
    elif kind == "mutate":
        if "value" in op["expr"]:
            op["expr"]["value"] = op["expr"].get("value", 0) + rnd.choice([-2, -1, 1, 2])
            return "tweak:mutate"
    return f"tweak:{kind or 'unknown'}:noop"


def _looks_like_date_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"dt", "date", "event_date", "timestamp"} or lowered.endswith(("_dt", "_date"))


def _looks_like_numeric_string_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"num_s", "number_text", "numeric_text"} or lowered.endswith(("_num_s", "_number_text", "_numeric_text"))


def _ensure_numeric_string_column(
    tables: list[TableData],
    available: list[str],
    rnd: random.Random,
    used_columns: set[str],
) -> str | None:
    numeric_strings = [
        column
        for column in available
        if _column_type(tables, column) == "str" and _looks_like_numeric_string_column(column)
        and _column_has_only_integer_strings(tables, column)
    ]
    if numeric_strings:
        return rnd.choice(numeric_strings)
    base_ints = [column.name for column in tables[0].columns if column.type == "int"]
    if not base_ints:
        return None
    source = "id" if "id" in base_ints else rnd.choice(base_ints)
    column = make_safe_output_name(f"{source}_num_s", used=used_columns)
    tables[0].columns.append(ColumnSpec(column, "str", nullable=True))
    for idx, row in enumerate(tables[0].rows):
        value = row.get(source)
        if value is None or idx % 5 == 0:
            row[column] = None
        else:
            row[column] = str(int(value))
    return column


def _column_has_only_integer_strings(tables: list[TableData], column: str) -> bool:
    seen = False
    for table in tables:
        for row in table.rows:
            if column not in row:
                continue
            value = row.get(column)
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                return False
            seen = True
    return seen


def _available_columns(tables: list[TableData], operations: list[dict[str, Any]]) -> list[str]:
    return _mutation_schema(tables, operations).available


def _column_type(tables: list[TableData], name: str) -> str:
    for table in tables:
        for column in table.columns:
            if column.name == name:
                return column.type
    return _fallback_column_type(name)


def _column_has_null(tables: list[TableData], name: str) -> bool:
    for table in tables:
        for column in table.columns:
            if column.name == name and column.nullable:
                return True
    return _base_table_column_has_null(tables, name)


def _literal_for_type(typ: str, rnd: random.Random) -> Any:
    catalog = _sample_catalog_literal(typ, rnd)
    if catalog is not None:
        _record_value_catalog_usage(catalog, column="", column_type=typ, purpose="literal")
        return catalog.value
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _sample_catalog_literal(
    typ: str,
    rnd: random.Random,
    *,
    column: str = "",
) -> CatalogEntry | None:
    context = _ACTIVE_VALUE_CATALOG_CONTEXT.get()
    if context is None:
        return None
    if rnd.random() >= VALUE_CATALOG_SAMPLE_PROBABILITY:
        return None
    column_type = "datetime" if typ == "str" and column and _looks_like_date_column(column) else typ
    return sample_catalog_entry(
        rnd,
        column_type,
        descriptor=context.descriptor,
        entry_scores=context.entry_scores,
    )


def _record_value_catalog_usage(
    entry: CatalogEntry,
    *,
    column: str,
    column_type: str,
    purpose: str,
) -> None:
    context = _ACTIVE_VALUE_CATALOG_CONTEXT.get()
    if context is None or not context.collect_usage:
        return
    payload = {
        "entry_id": entry.entry_id,
        "column": str(column),
        "column_type": str(column_type),
        "catalog_type": entry.column_type,
        "source_root_cause": entry.source_root_cause,
        "purpose": str(purpose),
    }
    if payload not in context.used_entries:
        context.used_entries.append(payload)


def _literal_list_for_type(typ: str, rnd: random.Random) -> list[Any]:
    if typ == "int":
        return rnd.sample([-10, -1, 0, 1, 2, 10], k=3)
    if typ == "float":
        return rnd.sample([-1.0, 0.0, 0.5, 1.0, 10.0], k=3)
    if typ == "bool":
        return rnd.sample([True, False], k=rnd.randint(1, 2))
    return rnd.sample(["", "alpha", "beta", "中文", "missing"], k=3)


MUTATION_OPERATORS: tuple[MutationOperator, ...] = (
    MutationOperator("value", _mutate_scalar_value),
    MutationOperator("nullify_value", _nullify_value),
    MutationOperator("duplicate_row", _duplicate_row),
    MutationOperator("drop_row", _drop_row),
    MutationOperator("shuffle_rows", _shuffle_rows),
    MutationOperator("append_op", _append_operation),
    MutationOperator("append_order_projection", _append_order_projection_probe),
    MutationOperator("append_truth_filter", _append_truth_filter_probe),
    MutationOperator("append_boolean_predicate_filter", _append_boolean_predicate_filter_probe),
    MutationOperator("append_range_filter", _append_range_filter_probe),
    MutationOperator("append_tuple_absence_filter", _append_tuple_absence_filter_probe),
    MutationOperator(
        "append_row_value_absence_filter",
        _append_row_value_absence_filter,
        semantic_family_affinity=("null_semantics", "join_membership"),
        semantic_signal_affinity=("row_value_absence_filter",),
    ),
    MutationOperator("append_running_sum", _append_running_sum_probe),
    MutationOperator("append_sortedness_check", _append_sortedness_check_probe),
    MutationOperator("append_random_case_probe", _append_random_case_probe),
    MutationOperator("append_group_quantile_probe", _append_group_quantile_probe),
    MutationOperator("append_scalar_subquery_probe", _append_scalar_subquery_probe),
    MutationOperator("append_window_avg_probe", _append_window_avg_probe),
    MutationOperator("append_struct_distinct_probe", _append_struct_distinct_probe),
    MutationOperator("append_bit_compare_probe", _append_bit_compare_probe),
    MutationOperator(
        "append_round_even_probe",
        _append_round_even_probe,
        divergence_affinity=("disagree_class:numeric", "mismatch:value"),
    ),
    MutationOperator(
        "append_series_rtruediv_probe",
        _append_series_rtruediv_probe,
        divergence_affinity=("disagree_class:numeric", "mismatch:value"),
    ),
    MutationOperator(
        "append_uint64_isin_probe",
        _append_uint64_isin_probe,
        divergence_affinity=("disagree_class:numeric", "mismatch:value"),
    ),
    MutationOperator(
        "append_tuple_anti_null_probe",
        _append_tuple_anti_null_probe,
        divergence_affinity=("disagree_class:null", "mismatch:value"),
    ),
    MutationOperator(
        "append_setop_all_duplicate_probe",
        _append_setop_all_duplicate_probe,
        divergence_affinity=("mismatch:row_count", "mismatch:value"),
    ),
    MutationOperator(
        "append_json_predicate_order_probe",
        _append_json_predicate_order_probe,
        divergence_affinity=("disagree_class:string", "mismatch:value"),
    ),
    MutationOperator(
        "append_sparse_mask_probe",
        _append_sparse_mask_probe,
        divergence_affinity=("disagree_class:bool", "mismatch:value"),
    ),
    MutationOperator(
        "append_float_wrap_probe",
        _append_float_wrap_probe,
        divergence_affinity=("disagree_class:numeric", "mismatch:value"),
    ),
    MutationOperator("append_index_bool_probe", _append_index_bool_probe),
    MutationOperator("append_empty_literal_groupby_probe", _append_empty_literal_groupby_probe),
    MutationOperator(
        "append_arrow_string_eq_sum_probe",
        _append_arrow_string_eq_sum_probe,
        divergence_affinity=("disagree_class:string", "mismatch:value"),
    ),
    MutationOperator(
        "append_arrow_timestamp_loc_slice_probe",
        _append_arrow_timestamp_loc_slice_probe,
        divergence_affinity=("disagree_class:string", "mismatch:value"),
    ),
    MutationOperator(
        "append_arrow_timestamp_index_attr_probe",
        _append_arrow_timestamp_index_attr_probe,
        divergence_affinity=("disagree_class:string", "mismatch:value"),
    ),
    MutationOperator("append_eval_inplace_alias_probe", _append_eval_inplace_alias_probe),
    MutationOperator(
        "append_bool_reduction_skipna_probe",
        _append_bool_reduction_skipna_probe,
        divergence_affinity=("disagree_class:bool", "mismatch:value"),
    ),
    MutationOperator("append_dataset_isin_all_match_probe", _append_dataset_isin_all_match_probe),
    MutationOperator(
        "append_run_end_null_compute_probe",
        _append_run_end_null_compute_probe,
        divergence_affinity=("disagree_class:null", "mismatch:value"),
    ),
    MutationOperator(
        "append_large_string_partition_probe",
        _append_large_string_partition_probe,
        divergence_affinity=("disagree_class:string", "mismatch:value"),
    ),
    MutationOperator("append_hash_pivot_wider_probe", _append_hash_pivot_wider_probe),
    MutationOperator("append_list_flatten_parent_indices_probe", _append_list_flatten_parent_indices_probe),
    MutationOperator("append_rolling_mean_by_null_count_probe", _append_rolling_mean_by_null_count_probe),
    MutationOperator("append_grouped_topk", _append_grouped_topk_probe),
    MutationOperator(
        "append_groupby_fractional_membership_filter",
        _append_groupby_fractional_membership_filter,
        semantic_family_affinity=("aggregation_cardinality", "join_membership", "type_coercion"),
        semantic_signal_affinity=("distinct_count_aggregation",),
    ),
    MutationOperator(
        "append_normalized_string_membership",
        _append_normalized_string_membership,
        semantic_family_affinity=("join_membership", "string_semantics"),
        semantic_signal_affinity=("normalized_string_membership_key", "chained_string_normalized_membership_key"),
    ),
    MutationOperator(
        "append_sql_distinct_null_topk",
        _append_sql_distinct_null_topk,
        semantic_family_affinity=("set_semantics", "topk_ordering", "null_semantics", "string_semantics"),
        semantic_signal_affinity=("sql_distinct_null_topk", "coalesced_distinct_topk"),
    ),
    MutationOperator(
        "append_left_join_coalesce_membership",
        _append_left_join_coalesce_membership,
        semantic_family_affinity=("join_membership", "null_semantics", "string_semantics"),
        semantic_signal_affinity=("left_join_coalesce_membership", "join_coalesce_membership"),
    ),
    MutationOperator(
        "append_left_join_case_membership",
        _append_left_join_case_membership,
        semantic_family_affinity=("join_membership", "conditional_semantics", "string_semantics"),
        semantic_signal_affinity=("left_join_case_when_membership", "join_case_when_membership"),
    ),
    MutationOperator(
        "append_empty_filter_global_aggregate",
        _append_empty_filter_global_aggregate,
        semantic_family_affinity=("aggregation_cardinality", "null_semantics"),
        semantic_signal_affinity=("filtered_global_aggregation",),
    ),
    MutationOperator(
        "append_sql_union_coalesce_distinct_topk",
        _append_sql_union_coalesce_distinct_topk,
        semantic_family_affinity=("set_semantics", "topk_ordering", "null_semantics"),
        semantic_signal_affinity=("union_coalesce_distinct_topk",),
    ),
    MutationOperator(
        "append_coalesce_sort_topk",
        _append_coalesce_sort_topk,
        semantic_family_affinity=("null_semantics", "topk_ordering", "type_coercion"),
        semantic_signal_affinity=("coalesce_sort_topk", "coalesced_topk"),
    ),
    MutationOperator(
        "append_boolean_membership_case_aggregate",
        _append_boolean_membership_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("boolean_membership_case_aggregation", "boolean_case_when_predicate"),
    ),
    MutationOperator(
        "append_left_join_boolean_case_aggregate",
        _append_left_join_boolean_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("left_join_boolean_case_aggregation", "boolean_case_when_aggregation"),
    ),
    MutationOperator(
        "append_left_join_boolean_coalesce_case_aggregate",
        _append_left_join_boolean_coalesce_case_aggregate,
        semantic_family_affinity=("boolean_logic", "join_membership", "conditional_semantics", "null_semantics"),
        semantic_signal_affinity=("left_join_boolean_coalesce_aggregation", "boolean_coalesce_case_aggregation"),
    ),
    MutationOperator(
        "append_boolean_antijoin_case_aggregate",
        _append_boolean_antijoin_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("boolean_membership_case_aggregation", "boolean_case_when_predicate"),
    ),
    MutationOperator(
        "append_left_join_boolean_coalesce_filter_aggregate",
        _append_left_join_boolean_coalesce_filter_aggregate,
        semantic_family_affinity=("boolean_logic", "join_membership", "null_semantics", "aggregation_cardinality"),
        semantic_signal_affinity=("left_join_boolean_coalesce_filter_aggregation", "boolean_coalesce_filter_aggregation"),
    ),
    MutationOperator(
        "append_numeric_text_boolean_antijoin_case_aggregate",
        _append_numeric_text_boolean_antijoin_case_aggregate,
        semantic_family_affinity=("type_coercion", "boolean_logic", "join_membership", "conditional_semantics"),
        semantic_signal_affinity=("numeric_text_cast_membership_aggregation", "boolean_membership_case_aggregation"),
    ),
    MutationOperator(
        "append_multi_key_membership_case_aggregate",
        _append_multi_key_membership_case_aggregate,
        semantic_family_affinity=("join_membership", "conditional_semantics", "aggregation_cardinality"),
        semantic_signal_affinity=("multi_key_membership_aggregation", "multi_key_semi_anti_join"),
    ),
    MutationOperator(
        "append_join_filter_groupby_topk",
        _append_join_filter_groupby_topk,
        semantic_family_affinity=("join_membership", "aggregation_cardinality", "topk_ordering"),
        semantic_signal_affinity=("join_filter_groupby", "multi_key_groupby_topk", "groupby_having_topk"),
    ),
    MutationOperator(
        "ir_swap_adjacent",
        apply_adjacent_independent_swap,
        semantic_family_affinity=("operation_rewrite", "ordering_semantics"),
        semantic_signal_affinity=("ir_adjacent_swap",),
        exploration_objective_affinity=("boundary_depth", "coverage_breadth"),
    ),
    MutationOperator(
        "ir_pushdown_filter",
        apply_filter_pushdown,
        semantic_family_affinity=("operation_rewrite", "predicate_logic", "join_membership"),
        semantic_signal_affinity=("ir_filter_pushdown",),
        exploration_objective_affinity=("boundary_depth", "cross_model_consistency"),
    ),
    MutationOperator(
        "ir_pull_filter_above_groupby",
        apply_filter_above_groupby,
        semantic_family_affinity=("operation_rewrite", "predicate_logic", "aggregation_cardinality"),
        semantic_signal_affinity=("ir_filter_above_groupby", "groupby_having_where_rewrite"),
        exploration_objective_affinity=("boundary_depth", "semantic_boundary"),
    ),
    MutationOperator(
        "ir_wrap_with_window",
        apply_wrap_with_window,
        semantic_family_affinity=("operation_rewrite", "ordering_semantics"),
        semantic_signal_affinity=("ir_window_wrap", "window_boundary"),
        exploration_objective_affinity=("coverage_breadth", "boundary_depth"),
    ),
    MutationOperator(
        "ir_splice_subtree",
        apply_subtree_splice,
        semantic_family_affinity=("operation_rewrite", "ordering_semantics"),
        semantic_signal_affinity=("ir_subtree_splice",),
        exploration_objective_affinity=("coverage_breadth", "boundary_depth"),
    ),
    MutationOperator(
        "ir_fold_redundant_op",
        apply_redundant_op_fold,
        semantic_family_affinity=("operation_rewrite", "ordering_semantics"),
        semantic_signal_affinity=("ir_redundant_fold",),
        exploration_objective_affinity=("state_space_minimization", "boundary_depth"),
    ),
    MutationOperator("drop_op", _drop_operation),
    MutationOperator("tweak_op", _tweak_random_operation),
    MutationOperator("shrink_drop_tail_op", shrink_drop_tail_op),
    MutationOperator("shrink_fold_redundant_op", shrink_fold_redundant_op),
    MutationOperator("shrink_merge_adjacent_filters", shrink_merge_adjacent_filters),
    MutationOperator("shrink_inline_single_use_mutate", shrink_inline_single_use_mutate),
)


def _with_inferred_objective_affinity(
    operators: tuple[MutationOperator, ...],
) -> tuple[MutationOperator, ...]:
    return tuple(_operator_with_inferred_objective_affinity(operator) for operator in operators)


def _operator_with_inferred_objective_affinity(operator: MutationOperator) -> MutationOperator:
    if operator.exploration_objective_affinity:
        return operator
    inferred = _infer_operator_objective_affinity(operator)
    if not inferred:
        return operator
    return MutationOperator(
        operator.name,
        operator.apply,
        semantic_family_affinity=operator.semantic_family_affinity,
        semantic_signal_affinity=operator.semantic_signal_affinity,
        exploration_objective_affinity=inferred,
        divergence_affinity=operator.divergence_affinity,
    )


def _infer_operator_objective_affinity(operator: MutationOperator) -> tuple[str, ...]:
    tokens = {
        operator.name,
        *operator.semantic_family_affinity,
        *operator.semantic_signal_affinity,
    }
    text = " ".join(tokens)
    objectives: list[str] = []

    def add(*values: str) -> None:
        for value in values:
            if value not in objectives:
                objectives.append(value)

    if operator.name in {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"}:
        add("representation_variance", "coverage_breadth")
    if any(fragment in text for fragment in ("filter", "boolean", "case", "predicate", "truth")):
        add("predicate_logic", "boundary_depth")
    if any(fragment in text for fragment in ("join", "membership", "groupby", "aggregate", "distinct", "topk", "union")):
        add("cross_model_consistency", "boundary_depth")
    if any(fragment in text for fragment in ("running", "window", "sortedness", "quantile", "rolling")):
        add("stateful_semantics")
    if any(fragment in text for fragment in ("cast", "string", "csv", "arrow", "sparse", "timestamp", "layout", "numeric")):
        add("representation_variance")
    if operator.name in {"append_op", "drop_op", "tweak_op"} or operator.name.endswith("_probe"):
        add("coverage_breadth")
    return tuple(objectives)


MUTATION_OPERATORS = _with_inferred_objective_affinity(MUTATION_OPERATORS)
PROBE_MUTATION_OPERATOR_NAMES = frozenset(
    operator.name
    for operator in MUTATION_OPERATORS
    if operator.name.endswith("_probe")
)
SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "append_grouped_topk",
        "append_running_sum",
        "append_sortedness_check",
        "append_tuple_absence_filter",
    }
)
ROOT_TARGETED_MUTATION_OPERATOR_NAMES = SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES
DISCOVERY_MUTATION_OPERATORS: tuple[MutationOperator, ...] = tuple(
    operator
    for operator in MUTATION_OPERATORS
    if operator.name not in PROBE_MUTATION_OPERATOR_NAMES
    and operator.name not in SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES
)
ALL_MUTATION_OPERATOR_PROFILES: Mapping[str, MutationOperator] = MappingProxyType(
    {operator.name: operator for operator in MUTATION_OPERATORS}
)
DISCOVERY_MUTATION_OPERATOR_PROFILES: Mapping[str, MutationOperator] = MappingProxyType(
    {operator.name: operator for operator in DISCOVERY_MUTATION_OPERATORS}
)
MUTATION_OPERATOR_NAMES = tuple(operator.name for operator in MUTATION_OPERATORS)
DISCOVERY_MUTATION_OPERATOR_NAMES = tuple(operator.name for operator in DISCOVERY_MUTATION_OPERATORS)
APPEND_ONLY_MUTATION_OPERATOR_NAMES = frozenset(
    operator.name
    for operator in MUTATION_OPERATORS
    if operator.name == "append_op" or operator.name.startswith("append_")
)
