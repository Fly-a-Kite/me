from __future__ import annotations

import cProfile
from collections import Counter
import json
import math
import pstats
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datadiff.adaptive_learning import AdaptiveLearningState, ContinualPriorityMemory, ExplorationMemory
from datadiff.finding_outcomes import (
    reward_signals_have_rewardable_finding,
    row_reward_signals,
)
from datadiff.util import dump_json, iter_jsonl, load_json, run_meta_path, utc_now

ADAPTIVE_BENCHMARK_SCHEMA_VERSION = "adaptive-benchmark-v1"
_LEARNING_BASELINE_VARIANT = "reward_signal_only"
_REPLAY_BENCHMARK_VARIANT_NAMES = ("reward_signal_only", "no_reward_model", "full_adaptive")
_REPLAY_SCOPE_SPECS = (
    ("generator_profile", "generator_profile_selection", ("profile", "action"), "selected_generator_profile", "profile_pool"),
    ("version_pair", "version_pair_selection", ("action",), "selected_version_pair", "action_pool"),
    ("semantic_objective", "semantic_objective_selection", ("action",), "selected_semantic_objective", "action_pool"),
    (
        "metamorphic_relation",
        "metamorphic_relation_selection",
        ("action",),
        "selected_metamorphic_relation",
        "action_pool",
    ),
)
_REPLAY_CONTEXT_PREFIXES = (
    "oracle_mode:",
    "guidance_strategy:",
    "candidate_pool:",
    "backend_count:",
    "backend:",
    "capability:",
    "target_family:",
    "target_layer:",
    "semantic_family:",
    "semantic_signal:",
    "matched_target:",
    "exploration_objective:",
    "guidance_target:",
    "version_pair:",
    "feedback:",
    "metamorphic:",
    "op:",
    "operation_root:",
    "operation_combo_root:",
)
_REPLAY_REWARD_INDEX_NAMES = (
    "exact_with_version",
    "exact",
    "focused_with_version",
    "focused",
    "action_with_version",
    "action",
    "scope",
    "global",
)


@dataclass(frozen=True, slots=True)
class LearningVariant:
    name: str
    enable_reward_model: bool
    enable_active_learning: bool
    enable_continual_learning: bool


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    scope: str
    action_pool: tuple[str, ...]
    choose_context_features: tuple[str, ...]
    record_context_features: tuple[str, ...]
    choose_version_id: str
    record_version_id: str
    chosen_action: str
    reward: float
    runtime_cost: float
    preflight_valid: bool
    fallback_used: bool
    false_positive: bool
    run_file: str
    case_index: int
    strategy: str


@dataclass(frozen=True, slots=True)
class ReplayRunSource:
    order_index: int
    run_file: Path
    source_refs: tuple[str, ...]
    meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayEstimate:
    reward: float
    runtime_cost: float
    preflight_valid: bool
    fallback_used: bool
    false_positive: bool
    matched_level: str


@dataclass(slots=True)
class ReplayAggregate:
    reward_total: float = 0.0
    runtime_cost_total: float = 0.0
    count: int = 0
    invalid_count: int = 0
    fallback_count: int = 0
    false_positive_count: int = 0

    def record(
        self,
        *,
        reward: float,
        runtime_cost: float,
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        self.reward_total += float(reward)
        self.runtime_cost_total += max(0.0, float(runtime_cost))
        self.count += 1
        if not preflight_valid:
            self.invalid_count += 1
        if fallback_used:
            self.fallback_count += 1
        if false_positive:
            self.false_positive_count += 1


def run_adaptive_benchmark(
    *,
    mode: str = "all",
    learning_rounds: int = 120,
    systems_iterations: int = 2000,
    profile_iterations: int = 1024,
    action_pool_size: int = 24,
    profile_top_n: int = 20,
    profile_output: Path | None = None,
    replay_run_files: list[Path] | None = None,
    replay_manifests: list[Path] | None = None,
) -> dict[str, Any]:
    normalized_mode = str(mode or "all").strip().lower()
    if normalized_mode not in {"all", "learning", "systems", "replay"}:
        raise ValueError(f"unsupported benchmark mode: {mode!r}")
    payload: dict[str, Any] = {
        "schema_version": ADAPTIVE_BENCHMARK_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": normalized_mode,
    }
    if normalized_mode in {"all", "learning"}:
        payload["learning_effectiveness"] = run_learning_effectiveness_benchmark(rounds=max(1, int(learning_rounds)))
    if normalized_mode in {"all", "systems"}:
        payload["systems_benchmark"] = run_systems_benchmark(
            iterations=max(8, int(systems_iterations)),
            profile_iterations=max(8, int(profile_iterations)),
            action_pool_size=max(2, int(action_pool_size)),
            profile_top_n=max(1, int(profile_top_n)),
            profile_output=profile_output,
        )
    replay_sources_present = bool(replay_run_files or replay_manifests)
    if normalized_mode == "replay" and not replay_sources_present:
        raise ValueError("adaptive benchmark replay mode requires at least one run log or manifest")
    if normalized_mode in {"all", "replay"} and replay_sources_present:
        payload["real_run_replay"] = run_real_run_replay_benchmark(
            replay_run_files=replay_run_files or [],
            replay_manifests=replay_manifests or [],
        )
    return payload


def run_learning_effectiveness_benchmark(*, rounds: int = 120) -> dict[str, Any]:
    variants = _learning_variants()
    scenario_rows = [
        _run_reward_model_context_split_scenario(rounds=rounds, variants=variants),
        _run_continual_priority_cold_start_scenario(rounds=rounds, variants=variants),
        _run_active_learning_cold_start_scenario(rounds=rounds, variants=variants),
    ]
    average_reward_by_variant = {
        variant.name: _mean(
            float(
                next(
                    row["average_reward"]
                    for row in scenario["variants"]
                    if row["name"] == variant.name
                )
            )
            for scenario in scenario_rows
        )
        for variant in variants
    }
    best_variant_name = max(
        average_reward_by_variant.items(),
        key=lambda item: (float(item[1]), item[0]),
    )[0]
    return {
        "summary": {
            "baseline_variant": _LEARNING_BASELINE_VARIANT,
            "scenario_count": len(scenario_rows),
            "variant_count": len(variants),
            "best_variant_by_average_reward": best_variant_name,
            "average_reward_by_variant": average_reward_by_variant,
        },
        "scenarios": scenario_rows,
    }


def run_systems_benchmark(
    *,
    iterations: int = 2000,
    profile_iterations: int = 1024,
    action_pool_size: int = 24,
    profile_top_n: int = 20,
    profile_output: Path | None = None,
) -> dict[str, Any]:
    scope = "generator_profile"
    actions = tuple(f"profile_{index:02d}" for index in range(action_pool_size))
    contexts = _benchmark_contexts()
    versions = ("v1", "v2", "v3")
    benchmark_rows = [
        _measure_microbenchmark(
            "state_choose",
            iterations=iterations,
            op_builder=lambda: _build_state_choose_op(
                _build_system_benchmark_state(action_pool_size=action_pool_size),
                scope=scope,
                actions=actions,
                contexts=contexts,
                versions=versions,
            ),
        ),
        _measure_microbenchmark(
            "bandit_rank_dense",
            iterations=iterations,
            op_builder=lambda: _build_rank_dense_op(
                _build_system_benchmark_state(action_pool_size=action_pool_size),
                scope=scope,
                actions=actions,
                contexts=contexts,
                versions=versions,
            ),
        ),
        _measure_microbenchmark(
            "state_rank_materialized",
            iterations=iterations,
            op_builder=lambda: _build_state_rank_op(
                _build_system_benchmark_state(action_pool_size=action_pool_size),
                scope=scope,
                actions=actions,
                contexts=contexts,
                versions=versions,
            ),
        ),
        _measure_microbenchmark(
            "state_score_action_materialized",
            iterations=iterations,
            op_builder=lambda: _build_state_score_action_op(
                _build_system_benchmark_state(action_pool_size=action_pool_size),
                scope=scope,
                actions=actions,
                contexts=contexts,
                versions=versions,
            ),
        ),
        _measure_microbenchmark(
            "state_record_outcome",
            iterations=iterations,
            op_builder=lambda: _build_state_record_op(
                _build_system_benchmark_state(action_pool_size=action_pool_size),
                scope=scope,
                actions=actions,
                contexts=contexts,
                versions=versions,
            ),
        ),
    ]
    benchmark_by_name = {row["name"]: row for row in benchmark_rows}
    profiler_payload = _run_systems_profiler(
        iterations=profile_iterations,
        action_pool_size=action_pool_size,
        profile_top_n=profile_top_n,
        output_path=profile_output,
    )
    return {
        "summary": {
            "iterations": int(iterations),
            "action_pool_size": int(action_pool_size),
            "materialized_rank_overhead_ratio": _ratio(
                benchmark_by_name["state_rank_materialized"]["mean_us"],
                benchmark_by_name["bandit_rank_dense"]["mean_us"],
            ),
            "materialized_score_overhead_ratio": _ratio(
                benchmark_by_name["state_score_action_materialized"]["mean_us"],
                benchmark_by_name["state_choose"]["mean_us"],
            ),
            "record_vs_choose_ratio": _ratio(
                benchmark_by_name["state_record_outcome"]["mean_us"],
                benchmark_by_name["state_choose"]["mean_us"],
            ),
            "process_max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "benchmarks": benchmark_rows,
        "profiler": profiler_payload,
    }


def write_adaptive_benchmark_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Adaptive Benchmark",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- mode: `{payload.get('mode', '')}`",
        "",
    ]
    learning = payload.get("learning_effectiveness")
    if isinstance(learning, dict):
        summary = learning.get("summary", {})
        lines.extend(
            [
                "## Learning Effectiveness",
                "",
                f"- baseline_variant: `{summary.get('baseline_variant', '')}`",
                f"- best_variant_by_average_reward: `{summary.get('best_variant_by_average_reward', '')}`",
                "",
            ]
        )
        for scenario in learning.get("scenarios", []) or []:
            lines.extend(
                [
                    f"### {scenario.get('scenario_id', 'unknown')}",
                    "",
                    str(scenario.get("description", "")),
                    "",
                    "| variant | cumulative_reward | average_reward | optimal_hit_rate | early_round_reward | first_step_optimal_rate |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in scenario.get("variants", []) or []:
                lines.append(
                    "| {name} | {cumulative_reward:.2f} | {average_reward:.2f} | {optimal_hit_rate:.3f} | {early_round_reward:.2f} | {first_step_optimal_rate:.3f} |".format(
                        name=str(row.get("name", "")),
                        cumulative_reward=float(row.get("cumulative_reward", 0.0) or 0.0),
                        average_reward=float(row.get("average_reward", 0.0) or 0.0),
                        optimal_hit_rate=float(row.get("optimal_hit_rate", 0.0) or 0.0),
                        early_round_reward=float(row.get("early_round_reward", 0.0) or 0.0),
                        first_step_optimal_rate=float(row.get("first_step_optimal_rate", 0.0) or 0.0),
                    )
                )
            lines.append("")
    replay = payload.get("real_run_replay")
    if isinstance(replay, dict):
        summary = replay.get("summary", {})
        lines.extend(
            [
                "## Real Run Replay",
                "",
                f"- run_file_count: `{summary.get('run_file_count', 0)}`",
                f"- manifest_count: `{summary.get('manifest_count', 0)}`",
                f"- event_count: `{summary.get('event_count', 0)}`",
                f"- baseline_variant: `{summary.get('baseline_variant', '')}`",
                f"- best_variant_by_average_reward: `{summary.get('best_variant_by_average_reward', '')}`",
                f"- recorded_average_reward: `{_fmt_float(summary.get('recorded_average_reward'))}`",
                "",
                "| variant | cumulative_reward | average_reward | policy_agreement_rate | invalid_rate | false_positive_rate | avg_runtime_cost |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in replay.get("variants", []) or []:
            lines.append(
                "| {name} | {cumulative_reward:.2f} | {average_reward:.2f} | {policy_agreement_rate:.3f} | {invalid_rate:.3f} | {false_positive_rate:.3f} | {avg_runtime_cost:.3f} |".format(
                    name=str(row.get("name", "")),
                    cumulative_reward=float(row.get("cumulative_reward", 0.0) or 0.0),
                    average_reward=float(row.get("average_reward", 0.0) or 0.0),
                    policy_agreement_rate=float(row.get("policy_agreement_rate", 0.0) or 0.0),
                    invalid_rate=float(row.get("invalid_rate", 0.0) or 0.0),
                    false_positive_rate=float(row.get("false_positive_rate", 0.0) or 0.0),
                    avg_runtime_cost=float(row.get("avg_runtime_cost", 0.0) or 0.0),
                )
            )
        lines.extend(
            [
                "",
                "| run_file | event_count | scope_counts |",
                "| --- | ---: | --- |",
            ]
        )
        for row in replay.get("sources", []) or []:
            scope_counts = ", ".join(
                f"{scope}:{count}"
                for scope, count in sorted((row.get("scope_counts", {}) or {}).items())
            ) or "none"
            lines.append(
                "| {run_file} | {event_count} | {scope_counts} |".format(
                    run_file=str(row.get("run_file", "")),
                    event_count=int(row.get("event_count", 0) or 0),
                    scope_counts=scope_counts,
                )
            )
        lines.append("")
    systems = payload.get("systems_benchmark")
    if isinstance(systems, dict):
        summary = systems.get("summary", {})
        lines.extend(
            [
                "## Systems Benchmark",
                "",
                f"- action_pool_size: `{summary.get('action_pool_size', '')}`",
                f"- materialized_rank_overhead_ratio: `{_fmt_float(summary.get('materialized_rank_overhead_ratio'))}`",
                f"- materialized_score_overhead_ratio: `{_fmt_float(summary.get('materialized_score_overhead_ratio'))}`",
                f"- record_vs_choose_ratio: `{_fmt_float(summary.get('record_vs_choose_ratio'))}`",
                "",
                "| benchmark | ops_per_s | mean_us | p50_us | p95_us | max_us |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in systems.get("benchmarks", []) or []:
            lines.append(
                "| {name} | {ops_per_s:.2f} | {mean_us:.2f} | {p50_us:.2f} | {p95_us:.2f} | {max_us:.2f} |".format(
                    name=str(row.get("name", "")),
                    ops_per_s=float(row.get("ops_per_s", 0.0) or 0.0),
                    mean_us=float(row.get("mean_us", 0.0) or 0.0),
                    p50_us=float(row.get("p50_us", 0.0) or 0.0),
                    p95_us=float(row.get("p95_us", 0.0) or 0.0),
                    max_us=float(row.get("max_us", 0.0) or 0.0),
                )
            )
        profiler = systems.get("profiler", {})
        lines.extend(["", "### Profiler", ""])
        if profiler.get("enabled"):
            lines.append(f"- profile_output: `{profiler.get('profile_output', '')}`")
            lines.append("")
            lines.append("| function | cumulative_s | self_s | calls | primitive_calls |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for row in profiler.get("top_functions", []) or []:
                lines.append(
                    "| {function} | {cumulative_s:.6f} | {self_s:.6f} | {calls} | {primitive_calls} |".format(
                        function=str(row.get("function", "")),
                        cumulative_s=float(row.get("cumulative_s", 0.0) or 0.0),
                        self_s=float(row.get("self_s", 0.0) or 0.0),
                        calls=int(row.get("calls", 0) or 0),
                        primitive_calls=int(row.get("primitive_calls", 0) or 0),
                    )
                )
        else:
            lines.append("- profiler disabled")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_real_run_replay_benchmark(
    *,
    replay_run_files: list[Path],
    replay_manifests: list[Path],
) -> dict[str, Any]:
    manifests = [Path(path) for path in replay_manifests]
    sources, seed_state, seed_summary = _resolve_replay_sources(
        replay_run_files=[Path(path) for path in replay_run_files],
        replay_manifests=manifests,
    )
    events: list[ReplayEvent] = []
    source_rows: list[dict[str, Any]] = []
    scope_event_counts: Counter[str] = Counter()
    for source in sources:
        run_events, row_count = _extract_replay_events(source)
        source_scope_counts = Counter(event.scope for event in run_events)
        scope_event_counts.update(source_scope_counts)
        source_rows.append(
            {
                "run_file": str(source.run_file),
                "source_refs": list(source.source_refs),
                "row_count": int(row_count),
                "event_count": len(run_events),
                "scope_counts": dict(sorted(source_scope_counts.items())),
            }
        )
        events.extend(run_events)
    if not events:
        raise ValueError("no replayable adaptive-learning events found in the provided run logs/manifests")
    variants = tuple(
        variant for variant in _learning_variants() if variant.name in _REPLAY_BENCHMARK_VARIANT_NAMES
    )
    actual_reward_total = sum(float(event.reward) for event in events)
    replay_index = _empty_replay_index()
    states = {variant.name: _clone_seed_state(seed_state) for variant in variants}
    metrics = {variant.name: _empty_replay_metrics(variant.name) for variant in variants}
    for event in events:
        for variant in variants:
            state = states[variant.name]
            chosen = state.choose(
                event.scope,
                event.action_pool,
                context_features=event.choose_context_features,
                version_id=event.choose_version_id,
                enable_active_learning=variant.enable_active_learning,
                enable_reward_model=variant.enable_reward_model,
                enable_continual_learning=variant.enable_continual_learning,
            )
            chosen_action = str(chosen or "").strip() or event.action_pool[0]
            if chosen_action not in event.action_pool:
                chosen_action = event.action_pool[0]
            estimate = _estimate_replay_outcome(replay_index, event, chosen_action)
            record_version_id = (
                chosen_action if event.scope == "version_pair" else event.record_version_id
            )
            state.record_outcome(
                event.scope,
                chosen_action,
                context_features=event.record_context_features,
                version_id=record_version_id,
                reward=estimate.reward,
                runtime_cost=estimate.runtime_cost,
                preflight_valid=estimate.preflight_valid,
                fallback_used=estimate.fallback_used,
                false_positive=estimate.false_positive,
                enable_reward_model=variant.enable_reward_model,
            )
            _update_replay_metrics(
                metrics[variant.name],
                event=event,
                chosen_action=chosen_action,
                estimate=estimate,
            )
        _record_replay_event(replay_index, event)
    variant_rows = [_finalize_replay_metrics(metrics[variant.name]) for variant in variants]
    baseline_row = next(
        (row for row in variant_rows if row["name"] == _LEARNING_BASELINE_VARIANT),
        None,
    )
    for row in variant_rows:
        row["delta_vs_baseline"] = (
            float(row["cumulative_reward"]) - float(baseline_row["cumulative_reward"])
            if baseline_row is not None
            else 0.0
        )
    average_reward_by_variant = {
        row["name"]: float(row.get("average_reward", 0.0) or 0.0)
        for row in variant_rows
    }
    best_variant_name = max(
        average_reward_by_variant.items(),
        key=lambda item: (item[1], item[0]),
    )[0]
    return {
        "schema_version": "adaptive-real-run-replay-v1",
        "seed_summary": seed_summary,
        "summary": {
            "baseline_variant": _LEARNING_BASELINE_VARIANT,
            "manifest_count": len(manifests),
            "run_file_count": len(sources),
            "event_count": len(events),
            "variant_count": len(variant_rows),
            "scope_event_counts": dict(sorted(scope_event_counts.items())),
            "best_variant_by_average_reward": best_variant_name,
            "average_reward_by_variant": average_reward_by_variant,
            "recorded_reward_total": actual_reward_total,
            "recorded_average_reward": actual_reward_total / len(events) if events else 0.0,
        },
        "sources": source_rows,
        "variants": variant_rows,
    }


def _resolve_replay_sources(
    *,
    replay_run_files: list[Path],
    replay_manifests: list[Path],
) -> tuple[list[ReplayRunSource], AdaptiveLearningState, dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    manifest_payloads: list[tuple[Path, dict[str, Any]]] = []

    def register_run(path_value: Path, *, ref: str) -> None:
        path = Path(path_value)
        key = str(path)
        if key not in seen:
            meta_path = run_meta_path(path)
            meta = load_json(meta_path) if meta_path.exists() else {}
            seen[key] = {
                "run_file": path,
                "source_refs": [ref],
                "meta": meta if isinstance(meta, dict) else {},
            }
            return
        refs = seen[key]["source_refs"]
        if ref not in refs:
            refs.append(ref)

    for run_file in replay_run_files:
        register_run(run_file, ref="direct_run_file")
    for manifest_path in replay_manifests:
        manifest = load_json(Path(manifest_path))
        manifest_payload = manifest if isinstance(manifest, dict) else {}
        manifest_payloads.append((Path(manifest_path), manifest_payload))
        for run in manifest_payload.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            run_file_value = str(run.get("run_file", "") or "").strip()
            if run_file_value:
                register_run(Path(run_file_value), ref=str(manifest_path))
    sources = [
        ReplayRunSource(
            order_index=index,
            run_file=item["run_file"],
            source_refs=tuple(item["source_refs"]),
            meta=dict(item["meta"]),
        )
        for index, item in enumerate(seen.values())
    ]
    if not sources:
        raise ValueError("adaptive benchmark replay requires at least one resolved run file")
    seed_state = AdaptiveLearningState()
    adaptive_learning_source = ""
    for source in sources:
        adaptive_learning = source.meta.get("adaptive_learning") if isinstance(source.meta, dict) else None
        if isinstance(adaptive_learning, dict):
            loaded = AdaptiveLearningState.from_state_dict(adaptive_learning)
            seed_state = AdaptiveLearningState(
                version_memory=loaded.version_memory,
                continual_priority_memory=loaded.continual_priority_memory,
                exploration_memory=loaded.exploration_memory,
            )
            adaptive_learning_source = str(source.run_file)
    for manifest_path, manifest in manifest_payloads:
        adaptive_learning = manifest.get("adaptive_learning")
        if isinstance(adaptive_learning, dict):
            loaded = AdaptiveLearningState.from_state_dict(adaptive_learning)
            seed_state = AdaptiveLearningState(
                version_memory=loaded.version_memory,
                continual_priority_memory=loaded.continual_priority_memory,
                exploration_memory=loaded.exploration_memory,
            )
            adaptive_learning_source = str(manifest_path)
    imported_seed_count = 0
    for source in sources:
        seed = source.meta.get("adaptive_learning_seed") if isinstance(source.meta, dict) else None
        if not isinstance(seed, dict):
            continue
        continual = seed.get("continual_priority_memory")
        if isinstance(continual, dict):
            seed_state.continual_priority_memory.merge(
                ContinualPriorityMemory.from_state_dict(continual)
            )
            imported_seed_count += 1
    for _, manifest in manifest_payloads:
        seed = manifest.get("adaptive_learning_seed")
        if not isinstance(seed, dict):
            continue
        continual = seed.get("continual_priority_memory")
        if isinstance(continual, dict):
            seed_state.continual_priority_memory.merge(
                ContinualPriorityMemory.from_state_dict(continual)
            )
            imported_seed_count += 1
    return sources, seed_state, {
        "adaptive_learning_source": adaptive_learning_source,
        "imported_continual_seed_count": imported_seed_count,
        "continual_priority_summary": seed_state.continual_priority_memory.summary(),
        "exploration_summary": seed_state.exploration_memory.summary(),
    }


def _extract_replay_events(source: ReplayRunSource) -> tuple[list[ReplayEvent], int]:
    meta = source.meta if isinstance(source.meta, dict) else {}
    meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    known_saturated_bug_families = _row_string_tuple(
        meta_config.get("known_saturated_bug_families", ())
    )
    rows: list[ReplayEvent] = []
    row_count = 0
    for fallback_index, row in enumerate(iter_jsonl(source.run_file)):
        row_count += 1
        if not isinstance(row, dict):
            continue
        reward_signals = row_reward_signals(
            row,
            known_saturated_bug_families=known_saturated_bug_families,
        )
        reward = _row_selection_reward(row, reward_signals=reward_signals)
        runtime_cost = _runtime_cost_signal(row)
        preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
        preflight_valid = bool(preflight.get("valid", True))
        fallback_used = bool(preflight.get("fallback_used", False))
        false_positive = bool(reward_signals.get("false_positive", False))
        case_learning_context = _row_string_tuple(row.get("case_learning_context", ()))
        selected_version_pair = _selected_version_pair(row, meta=meta)
        for scope, selection_key, action_keys, fallback_key, pool_key in _REPLAY_SCOPE_SPECS:
            selection = row.get(selection_key, {})
            if not isinstance(selection, dict):
                continue
            strategy = str(selection.get("strategy", "") or "").strip()
            if strategy not in {"contextual_bandit", "contextual_bandit_warmup"}:
                continue
            chosen_action = _selection_action(selection, action_keys, row.get(fallback_key, ""))
            action_pool = _row_string_tuple(selection.get(pool_key, selection.get("action_pool", ())))
            if chosen_action and chosen_action not in action_pool:
                action_pool = tuple(dict.fromkeys((*action_pool, chosen_action)))
            if len(action_pool) <= 1 or not chosen_action:
                continue
            choose_context_features, record_context_features = _replay_event_contexts(
                scope=scope,
                row=row,
                meta=meta,
                case_learning_context=case_learning_context,
            )
            choose_version_id, record_version_id = _replay_event_versions(
                scope=scope,
                meta=meta,
                row=row,
                selected_version_pair=selected_version_pair,
                chosen_action=chosen_action,
            )
            rows.append(
                ReplayEvent(
                    scope=scope,
                    action_pool=action_pool,
                    choose_context_features=choose_context_features,
                    record_context_features=record_context_features,
                    choose_version_id=choose_version_id,
                    record_version_id=record_version_id,
                    chosen_action=chosen_action,
                    reward=reward,
                    runtime_cost=runtime_cost,
                    preflight_valid=preflight_valid,
                    fallback_used=fallback_used,
                    false_positive=false_positive,
                    run_file=str(source.run_file),
                    case_index=int(row.get("case_index", fallback_index) or fallback_index),
                    strategy=strategy,
                )
            )
    return rows, row_count


def _replay_event_contexts(
    *,
    scope: str,
    row: dict[str, Any],
    meta: dict[str, Any],
    case_learning_context: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if scope == "generator_profile":
        context = _generator_profile_replay_context(row=row, meta=meta)
        return context, context
    if scope == "version_pair":
        return _version_pair_replay_context(meta=meta), case_learning_context
    return case_learning_context, case_learning_context


def _replay_event_versions(
    *,
    scope: str,
    meta: dict[str, Any],
    row: dict[str, Any],
    selected_version_pair: str,
    chosen_action: str,
) -> tuple[str, str]:
    default_pair = _default_version_pair(meta)
    if scope == "generator_profile":
        return "", ""
    if scope == "version_pair":
        choose_version_id = default_pair or selected_version_pair
        return choose_version_id, chosen_action
    return selected_version_pair or default_pair, selected_version_pair or default_pair


def _generator_profile_replay_context(*, row: dict[str, Any], meta: dict[str, Any]) -> tuple[str, ...]:
    meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    backends = _row_string_list(meta.get("backends", ()))
    target_specs = meta.get("targets", []) if isinstance(meta.get("targets", []), list) else []
    target_capabilities = _row_string_list(meta.get("common_capabilities", ()))
    candidate_pool = int(
        row.get(
            "candidate_pool_size",
            (meta.get("guidance", {}) if isinstance(meta.get("guidance", {}), dict) else {}).get(
                "candidate_pool",
                1,
            ),
        )
        or 1
    )
    features = [
        f"oracle_mode:{str(meta_config.get('oracle_mode', 'differential') or 'differential').strip()}",
        f"guidance_strategy:{str(meta_config.get('guidance_strategy', 'fixed') or 'fixed').strip()}",
        _bucket_feature("candidate_pool", candidate_pool, [(1, "single"), (4, "small"), (8, "medium")], "large"),
        _bucket_feature("backend_count", len(backends), [(0, "none"), (1, "single"), (3, "few")], "many"),
    ]
    if bool(meta_config.get("enable_metamorphic_oracle", False)):
        features.append("metamorphic:enabled")
    if bool(meta_config.get("enable_feedback", False)):
        features.append("feedback:enabled")
    for backend in backends[:8]:
        features.append(f"backend:{backend}")
    for target in target_specs[:8]:
        if not isinstance(target, dict):
            continue
        family = str(target.get("family", "") or "").strip()
        layer = str(target.get("layer", "") or "").strip()
        if family:
            features.append(f"target_family:{family}")
        if layer:
            features.append(f"target_layer:{layer}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    for family in _row_string_list(meta_config.get("semantic_focus_families", ()))[:8]:
        features.append(f"semantic_family:{family}")
    for signal in _row_string_list(meta_config.get("semantic_focus_signals", ()))[:8]:
        features.append(f"semantic_signal:{signal}")
    for target in _row_string_list(meta_config.get("guidance_targets", ()))[:8]:
        features.append(f"guidance_target:{target}")
    return _row_string_tuple(features)


def _version_pair_replay_context(*, meta: dict[str, Any]) -> tuple[str, ...]:
    meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    backends = _row_string_list(meta.get("backends", ()))
    target_capabilities = _row_string_list(meta.get("common_capabilities", ()))
    features = [
        f"oracle_mode:{str(meta_config.get('oracle_mode', 'differential') or 'differential').strip()}",
        f"guidance_strategy:{str(meta_config.get('guidance_strategy', 'fixed') or 'fixed').strip()}",
        _bucket_feature("backend_count", len(backends), [(0, "none"), (1, "single"), (3, "few")], "many"),
    ]
    if bool(meta_config.get("enable_feedback", False)):
        features.append("feedback:enabled")
    if bool(meta_config.get("enable_metamorphic_oracle", False)):
        features.append("metamorphic:enabled")
    for backend in backends[:8]:
        features.append(f"backend:{backend}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    for family in _row_string_list(meta_config.get("semantic_focus_families", ()))[:8]:
        features.append(f"semantic_family:{family}")
    for signal in _row_string_list(meta_config.get("semantic_focus_signals", ()))[:8]:
        features.append(f"semantic_signal:{signal}")
    return _row_string_tuple(features)


def _selected_version_pair(row: dict[str, Any], *, meta: dict[str, Any]) -> str:
    value = str(row.get("selected_version_pair", "") or "").strip()
    if value:
        return value
    return _default_version_pair(meta)


def _default_version_pair(meta: dict[str, Any]) -> str:
    meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    value = str(meta.get("version_pair", "") or meta_config.get("selected_version_pair", "") or "").strip()
    if value:
        return value
    target_version = str(
        meta.get("target_version", "") or meta_config.get("target_version", "") or ""
    ).strip()
    fixed_version = str(
        meta.get("fixed_version", "") or meta_config.get("fixed_version", "") or ""
    ).strip()
    if not target_version and not fixed_version:
        return ""
    return f"{target_version}->{fixed_version}".rstrip("->")


def _empty_replay_index() -> dict[str, dict[tuple[Any, ...], ReplayAggregate]]:
    return {name: {} for name in _REPLAY_REWARD_INDEX_NAMES}


def _record_replay_event(
    replay_index: dict[str, dict[tuple[Any, ...], ReplayAggregate]],
    event: ReplayEvent,
) -> None:
    focused_context = _focused_replay_context(event.record_context_features)
    keys = (
        ("exact_with_version", (event.scope, event.chosen_action, event.record_context_features, event.record_version_id)),
        ("exact", (event.scope, event.chosen_action, event.record_context_features)),
        ("focused_with_version", (event.scope, event.chosen_action, focused_context, event.record_version_id)),
        ("focused", (event.scope, event.chosen_action, focused_context)),
        ("action_with_version", (event.scope, event.chosen_action, event.record_version_id)),
        ("action", (event.scope, event.chosen_action)),
        ("scope", (event.scope,)),
        ("global", tuple()),
    )
    for level, key in keys:
        cell = replay_index[level].get(key)
        if cell is None:
            cell = ReplayAggregate()
            replay_index[level][key] = cell
        cell.record(
            reward=event.reward,
            runtime_cost=event.runtime_cost,
            preflight_valid=event.preflight_valid,
            fallback_used=event.fallback_used,
            false_positive=event.false_positive,
        )


def _estimate_replay_outcome(
    replay_index: dict[str, dict[tuple[Any, ...], ReplayAggregate]],
    event: ReplayEvent,
    chosen_action: str,
) -> ReplayEstimate:
    if chosen_action == event.chosen_action:
        return ReplayEstimate(
            reward=event.reward,
            runtime_cost=event.runtime_cost,
            preflight_valid=event.preflight_valid,
            fallback_used=event.fallback_used,
            false_positive=event.false_positive,
            matched_level="current_observed",
        )
    record_version_id = chosen_action if event.scope == "version_pair" else event.record_version_id
    focused_context = _focused_replay_context(event.record_context_features)
    keys = (
        ("exact_with_version", (event.scope, chosen_action, event.record_context_features, record_version_id)),
        ("exact", (event.scope, chosen_action, event.record_context_features)),
        ("focused_with_version", (event.scope, chosen_action, focused_context, record_version_id)),
        ("focused", (event.scope, chosen_action, focused_context)),
        ("action_with_version", (event.scope, chosen_action, record_version_id)),
        ("action", (event.scope, chosen_action)),
        ("scope", (event.scope,)),
        ("global", tuple()),
    )
    for level, key in keys:
        aggregate = replay_index[level].get(key)
        if aggregate is not None and aggregate.count > 0:
            return _estimate_from_aggregate(aggregate, matched_level=level)
    return ReplayEstimate(
        reward=0.0,
        runtime_cost=0.0,
        preflight_valid=True,
        fallback_used=False,
        false_positive=False,
        matched_level="none",
    )


def _estimate_from_aggregate(
    aggregate: ReplayAggregate,
    *,
    matched_level: str,
) -> ReplayEstimate:
    count = max(1, int(aggregate.count))
    invalid_rate = aggregate.invalid_count / count
    fallback_rate = aggregate.fallback_count / count
    false_positive_rate = aggregate.false_positive_count / count
    return ReplayEstimate(
        reward=aggregate.reward_total / count,
        runtime_cost=aggregate.runtime_cost_total / count,
        preflight_valid=invalid_rate < 0.5,
        fallback_used=fallback_rate >= 0.5,
        false_positive=false_positive_rate >= 0.5,
        matched_level=matched_level,
    )


def _empty_replay_metrics(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "event_count": 0,
        "cumulative_reward": 0.0,
        "policy_agreement_count": 0,
        "runtime_cost_total": 0.0,
        "invalid_count": 0,
        "fallback_count": 0,
        "false_positive_count": 0,
        "scope_counts": Counter(),
        "estimate_level_counts": Counter(),
    }


def _update_replay_metrics(
    metrics: dict[str, Any],
    *,
    event: ReplayEvent,
    chosen_action: str,
    estimate: ReplayEstimate,
) -> None:
    metrics["event_count"] += 1
    metrics["cumulative_reward"] += float(estimate.reward)
    metrics["runtime_cost_total"] += float(estimate.runtime_cost)
    metrics["scope_counts"][event.scope] += 1
    metrics["estimate_level_counts"][estimate.matched_level] += 1
    if chosen_action == event.chosen_action:
        metrics["policy_agreement_count"] += 1
    if not estimate.preflight_valid:
        metrics["invalid_count"] += 1
    if estimate.fallback_used:
        metrics["fallback_count"] += 1
    if estimate.false_positive:
        metrics["false_positive_count"] += 1


def _finalize_replay_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    event_count = int(metrics.get("event_count", 0) or 0)
    cumulative_reward = float(metrics.get("cumulative_reward", 0.0) or 0.0)
    runtime_cost_total = float(metrics.get("runtime_cost_total", 0.0) or 0.0)
    return {
        "name": str(metrics.get("name", "") or ""),
        "event_count": event_count,
        "cumulative_reward": cumulative_reward,
        "average_reward": cumulative_reward / event_count if event_count else 0.0,
        "policy_agreement_rate": (
            int(metrics.get("policy_agreement_count", 0) or 0) / event_count if event_count else 0.0
        ),
        "avg_runtime_cost": runtime_cost_total / event_count if event_count else 0.0,
        "invalid_rate": int(metrics.get("invalid_count", 0) or 0) / event_count if event_count else 0.0,
        "fallback_rate": int(metrics.get("fallback_count", 0) or 0) / event_count if event_count else 0.0,
        "false_positive_rate": (
            int(metrics.get("false_positive_count", 0) or 0) / event_count if event_count else 0.0
        ),
        "scope_counts": dict(sorted((metrics.get("scope_counts") or Counter()).items())),
        "estimate_level_counts": dict(
            sorted((metrics.get("estimate_level_counts") or Counter()).items())
        ),
    }


def _clone_seed_state(seed_state: AdaptiveLearningState) -> AdaptiveLearningState:
    return AdaptiveLearningState.from_state_dict(seed_state.to_state_dict())


def _focused_replay_context(features: tuple[str, ...]) -> tuple[str, ...]:
    filtered = [
        feature
        for feature in features
        if any(feature.startswith(prefix) for prefix in _REPLAY_CONTEXT_PREFIXES)
    ]
    return _row_string_tuple(filtered)


def _selection_action(selection: dict[str, Any], keys: tuple[str, ...], fallback: Any = "") -> str:
    for key in keys:
        text = str(selection.get(key, "") or "").strip()
        if text:
            return text
    return str(fallback or "").strip()


def _row_selection_reward(row: dict[str, Any], *, reward_signals: dict[str, Any]) -> float:
    candidate_bug = bool(reward_signals.get("candidate_bug", False))
    rewardable_semantic = bool(reward_signals.get("rewardable_semantic_divergence", False))
    false_positive = bool(reward_signals.get("false_positive", False))
    rewardable_finding_count = int(reward_signals.get("candidate_bug_count", 0) or 0) + int(
        reward_signals.get("semantic_divergence_needs_confirmation_count", 0) or 0
    )
    new_behavior = _reward_signals_have_rewardable_new_behavior(row, reward_signals=reward_signals)
    preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
    duration_ms = float(row.get("duration_ms", 0.0) or 0.0)
    throughput_hint = min(0.25, 1.0 / duration_ms) if duration_ms > 0.0 else 0.0
    return (
        (3.0 if candidate_bug else 0.0)
        + (0.7 if rewardable_semantic else 0.0)
        + (0.45 if new_behavior else 0.0)
        + min(0.4, 0.05 * rewardable_finding_count)
        + throughput_hint
        - (2.5 if false_positive else 0.0)
        - (0.5 if not bool(preflight.get("valid", True)) else 0.0)
        - (0.25 if bool(preflight.get("fallback_used", False)) else 0.0)
    )


def _reward_signals_have_rewardable_new_behavior(
    row: dict[str, Any],
    *,
    reward_signals: dict[str, Any],
) -> bool:
    if not bool(row.get("signal_new_behavior", row.get("is_new_behavior", False))):
        return False
    if not row.get("findings"):
        return True
    return reward_signals_have_rewardable_finding(reward_signals)


def _runtime_cost_signal(row: dict[str, Any]) -> float:
    duration_ms = float(row.get("duration_ms", 0.0) or 0.0)
    return min(2.0, max(0.0, duration_ms / 1000.0))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bucket_feature(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _row_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _row_string_tuple(values: Any) -> tuple[str, ...]:
    return tuple(_row_string_list(values))


def _learning_variants() -> tuple[LearningVariant, ...]:
    return (
        LearningVariant(
            name="reward_signal_only",
            enable_reward_model=False,
            enable_active_learning=False,
            enable_continual_learning=False,
        ),
        LearningVariant(
            name="no_reward_model",
            enable_reward_model=False,
            enable_active_learning=True,
            enable_continual_learning=True,
        ),
        LearningVariant(
            name="no_continual",
            enable_reward_model=True,
            enable_active_learning=True,
            enable_continual_learning=False,
        ),
        LearningVariant(
            name="no_active_learning",
            enable_reward_model=True,
            enable_active_learning=False,
            enable_continual_learning=True,
        ),
        LearningVariant(
            name="full_adaptive",
            enable_reward_model=True,
            enable_active_learning=True,
            enable_continual_learning=True,
        ),
    )


def _run_reward_model_context_split_scenario(
    *,
    rounds: int,
    variants: tuple[LearningVariant, ...],
) -> dict[str, Any]:
    rows = [
        _simulate_reward_model_context_split(variant, rounds=rounds)
        for variant in variants
    ]
    return _learning_scenario_payload(
        scenario_id="reward_model_context_split",
        description=(
            "Two actions alternate between opposite semantic contexts. "
            "Arm-level reward_signal alone conflates the contexts; the online reward model "
            "can use context features to separate them."
        ),
        focus="reward_model",
        baseline_variant=_LEARNING_BASELINE_VARIANT,
        rounds=rounds,
        variants=rows,
    )


def _run_continual_priority_cold_start_scenario(
    *,
    rounds: int,
    variants: tuple[LearningVariant, ...],
) -> dict[str, Any]:
    seed_state = AdaptiveLearningState()
    seed_state.ingest_continual_ledger(
        {
            "families": [
                {"family": "a_regression@engine", "status": "regression"},
                {"family": "z_fixed@engine", "status": "fixed"},
            ]
        }
    )
    rows = [
        _simulate_two_step_cold_start_episode(
            scenario_id="continual_priority_cold_start",
            variant=variant,
            rounds=rounds,
            seed_state=seed_state,
            scope="semantic_objective",
            actions=("semantic_family:z_fixed", "semantic_family:a_regression"),
            context_features=("target_suite:engine",),
            version_id="v2",
            optimal_action="semantic_family:a_regression",
            reward_if_optimal=3.0,
            reward_if_suboptimal=-1.0,
            bandit_weights={
                "exploration_weight": 0.0,
                "model_weight": 0.0,
                "uncertainty_weight": 0.0,
                "version_weight": 0.0,
                "continual_priority_weight": 1.0,
                "active_learning_weight": 0.0,
            },
        )
        for variant in variants
    ]
    return _learning_scenario_payload(
        scenario_id="continual_priority_cold_start",
        description=(
            "Repeated two-step cold starts with imported regression/fixed ledger feedback. "
            "Only continual-learning signals know which unseen semantic family is worth trying first."
        ),
        focus="continual_learning",
        baseline_variant=_LEARNING_BASELINE_VARIANT,
        rounds=rounds,
        variants=rows,
    )


def _run_active_learning_cold_start_scenario(
    *,
    rounds: int,
    variants: tuple[LearningVariant, ...],
) -> dict[str, Any]:
    seed_state = AdaptiveLearningState()
    for _ in range(8):
        seed_state.exploration_memory.record(
            scope="mutation_operator",
            action_id="known_operator",
            context_features=("target_suite:core", "semantic_family:join"),
            version_id="v1",
            reward=0.75,
        )
    rows = [
        _simulate_two_step_cold_start_episode(
            scenario_id="active_learning_cold_start",
            variant=variant,
            rounds=rounds,
            seed_state=seed_state,
            scope="mutation_operator",
            actions=("fresh_operator", "known_operator"),
            context_features=("target_suite:new_backend", "semantic_family:join"),
            version_id="v2",
            optimal_action="fresh_operator",
            reward_if_optimal=2.5,
            reward_if_suboptimal=0.0,
            bandit_weights={
                "exploration_weight": 0.0,
                "model_weight": 0.0,
                "uncertainty_weight": 0.0,
                "version_weight": 0.0,
                "continual_priority_weight": 0.0,
                "active_learning_weight": 1.0,
            },
        )
        for variant in variants
    ]
    return _learning_scenario_payload(
        scenario_id="active_learning_cold_start",
        description=(
            "Repeated two-step cold starts with exploration memory primed on a stale known operator. "
            "Active-learning uncertainty sampling should lift the fresh operator before arm history exists."
        ),
        focus="active_learning",
        baseline_variant=_LEARNING_BASELINE_VARIANT,
        rounds=rounds,
        variants=rows,
    )


def _simulate_reward_model_context_split(
    variant: LearningVariant,
    *,
    rounds: int,
) -> dict[str, Any]:
    state = AdaptiveLearningState()
    scope = "generator_profile"
    _configure_bandit_weights(
        state,
        scope,
        exploration_weight=0.20,
        model_weight=1.0,
        uncertainty_weight=0.35,
        version_weight=0.0,
        continual_priority_weight=0.0,
        active_learning_weight=0.0,
    )
    actions = ("profile_groupby", "profile_join")
    metrics = _empty_learning_metrics()
    for step in range(rounds):
        context_name = "join" if step % 2 == 0 else "groupby"
        optimal_action = "profile_join" if context_name == "join" else "profile_groupby"
        context_features = ("target_suite:core", f"semantic_family:{context_name}")
        chosen = state.choose(
            scope,
            actions,
            context_features=context_features,
            version_id="v1",
            enable_active_learning=variant.enable_active_learning,
            enable_reward_model=variant.enable_reward_model,
            enable_continual_learning=variant.enable_continual_learning,
        )
        reward = 3.0 if chosen == optimal_action else -1.0
        state.record_outcome(
            scope,
            chosen,
            context_features=context_features,
            version_id="v1",
            reward=reward,
            enable_reward_model=variant.enable_reward_model,
        )
        _update_learning_metrics(
            metrics,
            reward=reward,
            is_optimal=chosen == optimal_action,
            step_index=step,
            first_step=(step % 2 == 0),
            optimal_reward=3.0,
        )
    return _finalize_learning_metrics(variant.name, metrics, total_steps=rounds)


def _simulate_two_step_cold_start_episode(
    *,
    scenario_id: str,
    variant: LearningVariant,
    rounds: int,
    seed_state: AdaptiveLearningState,
    scope: str,
    actions: tuple[str, str],
    context_features: tuple[str, ...],
    version_id: str,
    optimal_action: str,
    reward_if_optimal: float,
    reward_if_suboptimal: float,
    bandit_weights: dict[str, float],
) -> dict[str, Any]:
    metrics = _empty_learning_metrics()
    total_steps = 0
    for episode in range(rounds):
        state = AdaptiveLearningState(
            continual_priority_memory=ContinualPriorityMemory.from_state_dict(
                seed_state.continual_priority_memory.to_state_dict()
            ),
            exploration_memory=ExplorationMemory.from_state_dict(
                seed_state.exploration_memory.to_state_dict()
            ),
        )
        _configure_bandit_weights(state, scope, **bandit_weights)
        for episode_step in range(2):
            chosen = state.choose(
                scope,
                actions,
                context_features=context_features,
                version_id=version_id,
                enable_active_learning=variant.enable_active_learning,
                enable_reward_model=variant.enable_reward_model,
                enable_continual_learning=variant.enable_continual_learning,
            )
            reward = reward_if_optimal if chosen == optimal_action else reward_if_suboptimal
            state.record_outcome(
                scope,
                chosen,
                context_features=context_features,
                version_id=version_id,
                reward=reward,
                enable_reward_model=variant.enable_reward_model,
            )
            _update_learning_metrics(
                metrics,
                reward=reward,
                is_optimal=chosen == optimal_action,
                step_index=total_steps,
                first_step=(episode_step == 0),
                optimal_reward=reward_if_optimal,
            )
            total_steps += 1
    return _finalize_learning_metrics(variant.name, metrics, total_steps=total_steps)


def _learning_scenario_payload(
    *,
    scenario_id: str,
    description: str,
    focus: str,
    baseline_variant: str,
    rounds: int,
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_row = next((row for row in variants if row["name"] == baseline_variant), None)
    return {
        "scenario_id": scenario_id,
        "description": description,
        "focus": focus,
        "rounds": int(rounds),
        "variants": variants,
        "baseline_variant": baseline_variant,
        "gains_vs_baseline": [
            {
                "name": row["name"],
                "delta_cumulative_reward": float(row["cumulative_reward"]) - float(baseline_row["cumulative_reward"]),
                "delta_optimal_hit_rate": float(row["optimal_hit_rate"]) - float(baseline_row["optimal_hit_rate"]),
                "delta_early_round_reward": float(row["early_round_reward"]) - float(baseline_row["early_round_reward"]),
                "delta_first_step_optimal_rate": float(row["first_step_optimal_rate"])
                - float(baseline_row["first_step_optimal_rate"]),
            }
            for row in variants
            if baseline_row is not None and row["name"] != baseline_variant
        ],
    }


def _empty_learning_metrics() -> dict[str, Any]:
    return {
        "cumulative_reward": 0.0,
        "optimal_hits": 0,
        "early_round_reward": 0.0,
        "first_step_hits": 0,
        "first_step_count": 0,
        "time_to_first_optimal": None,
        "cumulative_optimal_reward": 0.0,
    }


def _update_learning_metrics(
    metrics: dict[str, Any],
    *,
    reward: float,
    is_optimal: bool,
    step_index: int,
    first_step: bool,
    optimal_reward: float,
) -> None:
    metrics["cumulative_reward"] += float(reward)
    metrics["cumulative_optimal_reward"] += float(optimal_reward)
    if step_index < 8:
        metrics["early_round_reward"] += float(reward)
    if first_step:
        metrics["first_step_count"] += 1
        if is_optimal:
            metrics["first_step_hits"] += 1
    if is_optimal:
        metrics["optimal_hits"] += 1
        if metrics["time_to_first_optimal"] is None:
            metrics["time_to_first_optimal"] = int(step_index) + 1


def _finalize_learning_metrics(name: str, metrics: dict[str, Any], *, total_steps: int) -> dict[str, Any]:
    cumulative_reward = float(metrics["cumulative_reward"])
    cumulative_optimal_reward = float(metrics["cumulative_optimal_reward"])
    return {
        "name": name,
        "steps": int(total_steps),
        "cumulative_reward": cumulative_reward,
        "average_reward": cumulative_reward / total_steps if total_steps else 0.0,
        "optimal_hit_rate": metrics["optimal_hits"] / total_steps if total_steps else 0.0,
        "early_round_reward": float(metrics["early_round_reward"]),
        "first_step_optimal_rate": (
            metrics["first_step_hits"] / metrics["first_step_count"]
            if metrics["first_step_count"]
            else 0.0
        ),
        "time_to_first_optimal": metrics["time_to_first_optimal"],
        "regret": cumulative_optimal_reward - cumulative_reward,
    }


def _configure_bandit_weights(
    state: AdaptiveLearningState,
    scope: str,
    *,
    exploration_weight: float,
    model_weight: float,
    uncertainty_weight: float,
    version_weight: float,
    continual_priority_weight: float,
    active_learning_weight: float,
) -> None:
    bandit = state._bandit(scope)
    bandit.exploration_weight = float(exploration_weight)
    bandit.model_weight = float(model_weight)
    bandit.uncertainty_weight = float(uncertainty_weight)
    bandit.version_weight = float(version_weight)
    bandit.continual_priority_weight = float(continual_priority_weight)
    bandit.active_learning_weight = float(active_learning_weight)


def _build_system_benchmark_state(*, action_pool_size: int) -> AdaptiveLearningState:
    state = AdaptiveLearningState()
    scope = "generator_profile"
    actions = [f"profile_{index:02d}" for index in range(action_pool_size)]
    versions = ("v1", "v2", "v3")
    contexts = _benchmark_contexts()
    state.ingest_continual_ledger(
        {
            "families": [
                {"family": "join@engine", "status": "regression"},
                {"family": "groupby@engine", "status": "fixed"},
                {"family": "null@engine", "status": "persistent"},
            ]
        }
    )
    for index in range(max(128, action_pool_size * 10)):
        action = actions[index % len(actions)]
        context = contexts[index % len(contexts)]
        version = versions[index % len(versions)]
        reward = _system_reward(action, context, index)
        state.record_outcome(
            scope,
            action,
            context_features=context,
            version_id=version,
            reward=reward,
            runtime_cost=0.25 if index % 11 == 0 else 0.0,
            preflight_valid=index % 13 != 0,
            false_positive=index % 17 == 0,
        )
    return state


def _benchmark_contexts() -> tuple[tuple[str, ...], ...]:
    return (
        ("target_suite:core", "semantic_family:join"),
        ("target_suite:core", "semantic_family:groupby"),
        ("target_suite:engine", "semantic_family:null"),
        ("target_suite:fresh", "semantic_family:cast"),
    )


def _system_reward(action_id: str, context_features: tuple[str, ...], step: int) -> float:
    action_bucket = int(action_id.rsplit("_", 1)[-1]) % 4
    if "semantic_family:join" in context_features:
        return 2.0 if action_bucket == 1 else -0.5
    if "semantic_family:groupby" in context_features:
        return 2.0 if action_bucket == 0 else -0.5
    if "semantic_family:null" in context_features:
        return 1.5 if action_bucket in {2, 3} else -0.25
    return 1.0 if (action_bucket + step) % 2 == 0 else -0.25


def _build_state_choose_op(
    state: AdaptiveLearningState,
    *,
    scope: str,
    actions: tuple[str, ...],
    contexts: tuple[tuple[str, ...], ...],
    versions: tuple[str, ...],
) -> Callable[[int], None]:
    def run(iteration: int) -> None:
        state.choose(
            scope,
            actions,
            context_features=contexts[iteration % len(contexts)],
            version_id=versions[iteration % len(versions)],
        )

    return run


def _build_rank_dense_op(
    state: AdaptiveLearningState,
    *,
    scope: str,
    actions: tuple[str, ...],
    contexts: tuple[tuple[str, ...], ...],
    versions: tuple[str, ...],
) -> Callable[[int], None]:
    bandit = state._bandit(scope)

    def run(iteration: int) -> None:
        bandit.rank_dense(
            actions,
            scope=scope,
            context_features=contexts[iteration % len(contexts)],
            version_id=versions[iteration % len(versions)],
            version_memory=state.version_memory,
            continual_priority_memory=state.continual_priority_memory,
            exploration_memory=state.exploration_memory,
            enable_reward_model=True,
            enable_continual_learning=True,
        )

    return run


def _build_state_rank_op(
    state: AdaptiveLearningState,
    *,
    scope: str,
    actions: tuple[str, ...],
    contexts: tuple[tuple[str, ...], ...],
    versions: tuple[str, ...],
) -> Callable[[int], None]:
    def run(iteration: int) -> None:
        state.rank(
            scope,
            actions,
            context_features=contexts[iteration % len(contexts)],
            version_id=versions[iteration % len(versions)],
        )

    return run


def _build_state_score_action_op(
    state: AdaptiveLearningState,
    *,
    scope: str,
    actions: tuple[str, ...],
    contexts: tuple[tuple[str, ...], ...],
    versions: tuple[str, ...],
) -> Callable[[int], None]:
    def run(iteration: int) -> None:
        action = actions[iteration % len(actions)]
        state.score_action(
            scope,
            action,
            context_features=contexts[iteration % len(contexts)],
            version_id=versions[iteration % len(versions)],
        )

    return run


def _build_state_record_op(
    state: AdaptiveLearningState,
    *,
    scope: str,
    actions: tuple[str, ...],
    contexts: tuple[tuple[str, ...], ...],
    versions: tuple[str, ...],
) -> Callable[[int], None]:
    def run(iteration: int) -> None:
        action = actions[iteration % len(actions)]
        context = contexts[iteration % len(contexts)]
        state.record_outcome(
            scope,
            action,
            context_features=context,
            version_id=versions[iteration % len(versions)],
            reward=_system_reward(action, context, iteration),
            runtime_cost=0.15 if iteration % 19 == 0 else 0.0,
            preflight_valid=iteration % 23 != 0,
            false_positive=iteration % 29 == 0,
        )

    return run


def _measure_microbenchmark(
    name: str,
    *,
    iterations: int,
    op_builder: Callable[[], Callable[[int], None]],
) -> dict[str, Any]:
    operation = op_builder()
    warmup = min(32, max(4, iterations // 8))
    for index in range(warmup):
        operation(index)
    samples_ns: list[int] = []
    started_ns = time.perf_counter_ns()
    for iteration in range(iterations):
        tick_ns = time.perf_counter_ns()
        operation(iteration)
        samples_ns.append(time.perf_counter_ns() - tick_ns)
    total_ns = time.perf_counter_ns() - started_ns
    sorted_samples = sorted(samples_ns)
    total_s = total_ns / 1_000_000_000.0
    return {
        "name": name,
        "iterations": int(iterations),
        "total_s": total_s,
        "ops_per_s": iterations / total_s if total_s > 0.0 else 0.0,
        "mean_us": (sum(samples_ns) / iterations) / 1_000.0 if iterations else 0.0,
        "p50_us": _percentile_us(sorted_samples, 0.50),
        "p95_us": _percentile_us(sorted_samples, 0.95),
        "max_us": (sorted_samples[-1] / 1_000.0) if sorted_samples else 0.0,
    }


def _run_systems_profiler(
    *,
    iterations: int,
    action_pool_size: int,
    profile_top_n: int,
    output_path: Path | None,
) -> dict[str, Any]:
    if output_path is None:
        return {"enabled": False, "top_functions": []}
    state = _build_system_benchmark_state(action_pool_size=action_pool_size)
    actions = tuple(f"profile_{index:02d}" for index in range(action_pool_size))
    contexts = _benchmark_contexts()
    versions = ("v1", "v2", "v3")

    def workload() -> None:
        for iteration in range(iterations):
            context = contexts[iteration % len(contexts)]
            version = versions[iteration % len(versions)]
            chosen = state.choose(
                "generator_profile",
                actions,
                context_features=context,
                version_id=version,
            )
            state.record_outcome(
                "generator_profile",
                chosen,
                context_features=context,
                version_id=version,
                reward=_system_reward(chosen, context, iteration),
                runtime_cost=0.10 if iteration % 19 == 0 else 0.0,
                preflight_valid=iteration % 23 != 0,
                false_positive=iteration % 29 == 0,
            )
            if iteration % 16 == 0:
                state._bandit("generator_profile").rank_dense(
                    actions,
                    scope="generator_profile",
                    context_features=context,
                    version_id=version,
                    version_memory=state.version_memory,
                    continual_priority_memory=state.continual_priority_memory,
                    exploration_memory=state.exploration_memory,
                    enable_reward_model=True,
                    enable_continual_learning=True,
                )
            if iteration % 32 == 0:
                state.rank(
                    "generator_profile",
                    actions,
                    context_features=context,
                    version_id=version,
                )

    profile = cProfile.Profile()
    profile.runcall(workload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(output_path))
    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    top_functions = []
    for function_id in stats.fcn_list[: max(1, profile_top_n)]:
        primitive_calls, total_calls, self_s, cumulative_s, _ = stats.stats[function_id]
        top_functions.append(
            {
                "function": _format_profile_function(function_id),
                "primitive_calls": int(primitive_calls),
                "calls": int(total_calls),
                "self_s": float(self_s),
                "cumulative_s": float(cumulative_s),
            }
        )
    return {
        "enabled": True,
        "iterations": int(iterations),
        "profile_output": str(output_path),
        "top_functions": top_functions,
    }


def _percentile_us(sorted_samples_ns: list[int], fraction: float) -> float:
    if not sorted_samples_ns:
        return 0.0
    index = int(math.ceil(fraction * len(sorted_samples_ns))) - 1
    index = max(0, min(len(sorted_samples_ns) - 1, index))
    return sorted_samples_ns[index] / 1_000.0


def _format_profile_function(function_id: tuple[str, int, str]) -> str:
    filename, lineno, function = function_id
    return f"{filename}:{lineno}({function})"


def _mean(values: list[float] | Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _ratio(numerator: Any, denominator: Any) -> float:
    denom = float(denominator or 0.0)
    if denom <= 0.0:
        return 0.0
    return float(numerator or 0.0) / denom


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"
