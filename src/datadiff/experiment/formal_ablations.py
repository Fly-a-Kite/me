"""Declared P2 single-factor ablations for the next architecture.

Each factor is an independent ``ExperimentPlan``. All arms consume immutable
cases from one frozen corpus; scheduler arms see a larger frozen candidate
reservoir but execute the same preregistered case budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from datadiff.ccs_ablation import freeze_case_corpus, validate_frozen_case_corpus
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.evidence import ContentAddressedSidecar
from datadiff.experiment.campaign import (
    CampaignRunner,
    _compact_case_row,
    _frozen_corpus_metrics,
    _fuzz_config_for_method,
)
from datadiff.experiment.plan import ExperimentPlan, MethodSpec, SeedBlock, ShardSpec
from datadiff.experiment_manifest import stable_digest
from datadiff.run_event_evidence import event_evidence_dir
from datadiff.runner import run_fuzz
from datadiff.util import dump_json, load_json, read_jsonl, run_meta_path, utc_now


FORMAL_ABLATION_FACTORS = (
    "evidence",
    "parallel",
    "scheduler",
    "grouping",
    "logging",
)
FORMAL_ABLATION_SCHEMA_VERSION = "next-architecture-formal-ablations-v1"
_CORPUS_PROFILES = (
    "null_predicate_filter",
    "null_groupby_topk",
    "skewed_join_multiplicity",
    "filter_null_agg_topk",
)
_COMMON_METHOD_OVERRIDES = {
    "execution_mode": "cartesian",
    "selector_mode": "all",
    "node_budget": None,
    "cache_mode": "disabled",
    "confirmation_mode": "none",
}
_GATE_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "evidence": (("io_bytes", "decrease"),),
    "parallel": (("wall_ms", "decrease"),),
    "scheduler": (("coverage_cells", "increase"),),
    "grouping": (("root_group_collision_count", "decrease"),),
    "logging": (
        ("io_bytes", "decrease"),
        ("evidence_completeness_rate", "increase"),
    ),
}


@dataclass(frozen=True, slots=True)
class FormalCorpusLayout:
    corpus: dict[str, Any]
    standard_blocks: tuple[SeedBlock, ...]
    scheduler_blocks: tuple[SeedBlock, ...]
    cases_per_block: int
    scheduler_candidate_pool: int


def build_formal_corpus(
    *,
    root_seed: int,
    block_count: int,
    cases_per_block: int,
    scheduler_candidate_pool: int,
) -> FormalCorpusLayout:
    """Freeze deterministic, profile-diverse candidate reservoirs."""

    block_count = max(1, int(block_count))
    cases_per_block = max(1, int(cases_per_block))
    scheduler_candidate_pool = max(2, int(scheduler_candidate_pool))
    cases: list[Case] = []
    standard_blocks: list[SeedBlock] = []
    scheduler_blocks: list[SeedBlock] = []
    reservoir_size = cases_per_block * scheduler_candidate_pool
    for block_index in range(block_count):
        block_seed = int(root_seed) + block_index * 10_000
        first_index = len(cases)
        for offset in range(reservoir_size):
            profile = _CORPUS_PROFILES[offset % len(_CORPUS_PROFILES)]
            cases.append(generate_case(block_seed + offset, profile=profile))
        reservoir_indices = tuple(range(first_index, first_index + reservoir_size))
        standard_blocks.append(
            SeedBlock(
                block_id=f"block-{block_index:02d}",
                root_seed=block_seed,
                case_indices=reservoir_indices[:cases_per_block],
            )
        )
        scheduler_blocks.append(
            SeedBlock(
                block_id=f"block-{block_index:02d}",
                root_seed=block_seed,
                case_indices=reservoir_indices,
            )
        )
    corpus = freeze_case_corpus(
        cases,
        provenance={
            "kind": "next_architecture_formal_ablation_corpus",
            "root_seed": int(root_seed),
            "block_count": block_count,
            "cases_per_block": cases_per_block,
            "scheduler_candidate_pool": scheduler_candidate_pool,
            "profiles": list(_CORPUS_PROFILES),
        },
    )
    return FormalCorpusLayout(
        corpus=corpus,
        standard_blocks=tuple(standard_blocks),
        scheduler_blocks=tuple(scheduler_blocks),
        cases_per_block=cases_per_block,
        scheduler_candidate_pool=scheduler_candidate_pool,
    )


def build_formal_ablation_plans(
    layout: FormalCorpusLayout,
    *,
    targets: Sequence[str],
    replicates: int = 3,
) -> dict[str, ExperimentPlan]:
    targets = tuple(str(item) for item in targets if str(item))
    if len(targets) < 2:
        raise ValueError("formal ablations require at least two targets")
    replicates = max(3, int(replicates))
    plans: dict[str, ExperimentPlan] = {}
    for factor in FORMAL_ABLATION_FACTORS:
        factor_targets = targets
        if factor == "logging":
            normal_targets = tuple(
                target for target in targets if not target.startswith("buggy_")
            )
            if len(normal_targets) >= 2:
                factor_targets = normal_targets
        plans[factor] = _single_factor_plan(
            factor,
            layout,
            targets=factor_targets,
            replicates=replicates,
        )
    return plans


def _base_config(
    *,
    candidate_pool: int = 1,
    coverage_debt: bool = False,
    log_level: str = "full",
    event_evidence: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        method_arm="p5_physical_plan",
        method_arm_overrides=dict(_COMMON_METHOD_OVERRIDES),
        enable_artifact=False,
        enable_reducer=False,
        candidate_recheck_count=0,
        enable_feedback=True,
        enable_replay_bug=True,
        persist_feedback_corpus=False,
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=False,
        enable_backend_sampling=False,
        enable_coordinator_coverage_debt=coverage_debt,
        coordinator_candidate_pool=candidate_pool,
        fresh_generation_min_share=1.0,
        feedback_mutation_max_share=0.0,
        enable_semantic_metamorphic_source=False,
        enable_known_regression_source=False,
        guidance_strategy="random",
        guidance_candidate_pool=1,
        enable_lhs_seeding=False,
        enable_metamorphic_oracle=False,
        log_level=log_level,
        enable_event_evidence=event_evidence,
    )


def _base_method(
    factor: str,
    layout: FormalCorpusLayout,
    *,
    targets: tuple[str, ...],
) -> MethodSpec:
    scheduler_factor = factor == "scheduler"
    logging_factor = factor == "logging"
    config = _base_config(
        candidate_pool=(layout.scheduler_candidate_pool if scheduler_factor else 1),
        coverage_debt=False,
        log_level="full",
        event_evidence=not logging_factor,
    )
    return MethodSpec(
        method_id=f"{factor}_control",
        parent_method_id="",
        changed_dimensions=(),
        generation={
            "corpus": "next-architecture-formal-frozen-v1",
            "seed_assignment": "frozen_manifest",
            "candidate_reservoir": "preregistered",
        },
        scheduling={
            "scheduler": "legacy-utility-feedback-v1",
            "coverage_debt": False,
            "candidate_pool": layout.scheduler_candidate_pool if scheduler_factor else 1,
            "executed_cases_per_block": layout.cases_per_block,
            "fresh_only": True,
        },
        execution={
            "topology": "same-case-serial",
            "workers": 1,
            "targets": list(targets),
            "fuzz_config": config.to_dict(),
        },
        evidence={
            "plan_collection": "full",
            "runtime_log": "legacy-full-row" if logging_factor else "manifest-sidecar",
            "raw_archive": "content-addressed-neutral-audit-archive",
            "rechecks": 0,
        },
        semantics={
            "normalizer": "lossless-v2",
            "oracle": "datadiff-oracle-v2",
            "capability_contract": "target-capability-model-v1",
            "causal_grouping": (
                "broad-semantic-tag-v1"
                if factor == "grouping"
                else "datadiff-causal-signature-v1"
            ),
        },
        arm_kind="research_control",
    )


def _single_factor_plan(
    factor: str,
    layout: FormalCorpusLayout,
    *,
    targets: tuple[str, ...],
    replicates: int,
) -> ExperimentPlan:
    if factor not in FORMAL_ABLATION_FACTORS:
        raise ValueError(f"unknown formal ablation factor: {factor}")
    control = _base_method(factor, layout, targets=targets)
    if factor == "evidence":
        evidence = dict(control.evidence)
        evidence.update(
            {
                "plan_collection": "tiered",
                "fuzz_config_overrides": {
                    "method_arm_overrides": {
                        **_COMMON_METHOD_OVERRIDES,
                        "plan_collection_mode": "tiered",
                    }
                },
            }
        )
        treatment = replace(
            control,
            method_id="evidence_tiered",
            parent_method_id=control.method_id,
            changed_dimensions=("evidence",),
            evidence=evidence,
            arm_kind="candidate",
        )
        primary = (
            "candidate_count",
            "independent_roots",
            "wall_ms",
            "cpu_ms",
            "io_bytes",
            "evidence_completeness_rate",
            "full_plan_evidence_rate",
            "plan_fingerprint_rate",
        )
    elif factor == "parallel":
        execution = dict(control.execution)
        execution.update(
            {
                "topology": "same-case-all-backends-parallel",
                "fuzz_config_overrides": {"enable_parallel_backend_execution": True},
            }
        )
        treatment = replace(
            control,
            method_id="parallel_all_backends",
            parent_method_id=control.method_id,
            changed_dimensions=("execution",),
            execution=execution,
        )
        primary = ("wall_ms", "cpu_ms", "rss_mib", "io_bytes")
    elif factor == "scheduler":
        scheduling = dict(control.scheduling)
        scheduling.update(
            {
                "scheduler": "coverage-debt-v1",
                "coverage_debt": True,
                "fuzz_config_overrides": {
                    "enable_coordinator_coverage_debt": True,
                },
            }
        )
        treatment = replace(
            control,
            method_id="scheduler_coverage_debt",
            parent_method_id=control.method_id,
            changed_dimensions=("scheduling",),
            scheduling=scheduling,
        )
        primary = (
            "independent_roots",
            "coverage_cells",
            "coverage_debt",
            "wall_ms",
            "cpu_ms",
        )
    elif factor == "grouping":
        semantics = dict(control.semantics)
        semantics["causal_grouping"] = "datadiff-causal-signature-v1"
        treatment = replace(
            control,
            method_id="grouping_causal_signature",
            parent_method_id=control.method_id,
            changed_dimensions=("semantics",),
            semantics=semantics,
        )
        primary = (
            "independent_roots",
            "root_group_collision_count",
            "broad_merged_causal_root_count",
            "wall_ms",
        )
    else:
        evidence = dict(control.evidence)
        evidence.update(
            {
                "runtime_log": "minimal-manifest-sidecar",
                "fuzz_config_overrides": {
                    "log_level": "minimal",
                    "enable_event_evidence": True,
                },
            }
        )
        treatment = replace(
            control,
            method_id="logging_manifest_sidecar",
            parent_method_id=control.method_id,
            changed_dimensions=("evidence",),
            evidence=evidence,
        )
        primary = (
            "io_bytes",
            "analysis_ms",
            "evidence_completeness_rate",
            "recovery_success_rate",
            "wall_ms",
        )
    seed_blocks = layout.scheduler_blocks if factor == "scheduler" else layout.standard_blocks
    return ExperimentPlan(
        plan_id=f"next-architecture-{factor}-single-factor-v1",
        design="single_factor_ablation",
        methods=(control, treatment),
        seed_blocks=seed_blocks,
        target_shards=(targets,),
        replicates=replicates,
        corpus_manifest_digest=str(layout.corpus["corpus_digest"]),
        resource_budget={
            "executed_cases_per_shard": layout.cases_per_block,
            "candidate_pool": layout.scheduler_candidate_pool if factor == "scheduler" else 1,
            "workers": 1,
            "target_count": len(targets),
        },
        stopping_rule={
            "kind": "complete_declared_matrix",
            "max_shards": 2 * replicates * len(seed_blocks),
        },
        primary_metrics=primary,
        secondary_metrics=(
            "candidate_count",
            "false_positive_rate",
            "undecided_rate",
            "harness_error_rate",
            "run_log_bytes",
            "sidecar_bytes",
        ),
        exclusion_rules=(
            "warmup is forbidden",
            "known replay is forbidden",
            "seeded fault detections are sensitivity evidence, not independent roots",
            "failed or incomplete shards are excluded and make the plan incomplete",
        ),
    )


class FrozenRunLoopShardExecutor:
    """Execute a run-loop shard with an immutable case provider."""

    def __init__(
        self,
        corpus: Mapping[str, Any],
        *,
        artifact_root: str | Path,
        replay_command: str,
    ) -> None:
        validate_frozen_case_corpus(corpus)
        self.corpus = dict(corpus)
        self.artifact_root = Path(artifact_root)
        self.replay_command = replay_command
        self.archive_root = self.artifact_root / "raw_archive"
        self.archive = ContentAddressedSidecar(self.archive_root)

    def __call__(self, shard: ShardSpec, method: MethodSpec) -> Mapping[str, Any]:
        config = _fuzz_config_for_method(method)
        case_payloads = self.corpus.get("cases", ())
        frozen_cases = [
            Case.from_dict(dict(case_payloads[index]))
            for index in shard.seed_block.case_indices
        ]
        by_seed = {case.seed: case for case in frozen_cases}
        if len(by_seed) != len(frozen_cases):
            raise ValueError("formal shard requires unique frozen case seeds")
        requested_seeds: list[int] = []

        def frozen_provider(seed: int, **_kwargs: Any) -> Case:
            requested_seeds.append(int(seed))
            case = by_seed.get(int(seed))
            if case is None:
                raise ValueError(f"seed {seed} is outside the preregistered shard reservoir")
            return Case.from_dict(case.to_dict())

        executed_cases = int(
            method.scheduling.get("executed_cases_per_block", len(frozen_cases))
            or len(frozen_cases)
        )
        shard_root = self.artifact_root / "run_shards" / shard.shard_id
        run_file = run_fuzz(
            cases=executed_cases,
            seed=shard.seed_block.root_seed,
            backends=list(shard.target_shard),
            config=config,
            runs_dir=shard_root / "runs",
            corpus_dir=shard_root / "corpus",
            generate_case_fn=frozen_provider,
        )
        rows = read_jsonl(run_file)
        meta = load_json(run_meta_path(run_file))
        if len(rows) != executed_cases or int(meta.get("executed_cases", 0) or 0) != executed_cases:
            raise RuntimeError("formal shard did not complete its preregistered case budget")
        selected_case_ids = [
            str((row.get("case", {}) or {}).get("case_id", "") or "")
            for row in rows
        ]
        allowed_case_ids = {case.case_id for case in frozen_cases}
        if any(case_id not in allowed_case_ids for case_id in selected_case_ids):
            raise RuntimeError("run loop selected a case outside the frozen reservoir")
        metrics = _frozen_corpus_metrics(
            rows,
            coordinator_snapshot=(
                meta.get("coordinator", {})
                if isinstance(meta.get("coordinator", {}), Mapping)
                else {}
            ),
            grouping_mode=str(
                method.semantics.get(
                    "causal_grouping",
                    "datadiff-causal-signature-v1",
                )
            ),
        )
        metrics.update(_plan_evidence_metrics(rows))
        recovery = _recovery_metrics(run_file, rows, event_enabled=config.enable_event_evidence)
        artifact_metrics = _runtime_artifact_metrics(run_file)
        metrics.update(
            {
                "wall_ms": _nested_number(
                    meta,
                    "wall_time_profile",
                    "totals_ms",
                    "total_wall_ms",
                ),
                "cpu_ms": _nested_number(
                    meta,
                    "process_cpu_profile",
                    "totals_ms",
                    "total_process_cpu_ms",
                ),
                "rss_mib": _row_max_rss(rows),
                "backend_io_bytes": metrics.get("io_bytes", 0.0),
                "io_bytes": float(metrics.get("io_bytes", 0.0) or 0.0)
                + artifact_metrics["artifact_bytes"],
                "analysis_ms": _analysis_elapsed_ms(
                    run_file,
                    include_events=config.enable_event_evidence,
                ),
                **artifact_metrics,
                **recovery,
            }
        )
        archived_rows = [_compact_case_row(row, sidecars=self.archive) for row in rows]
        archival_ref_count = sum(
            len(row.get("raw_result_refs", {}) or {})
            for row in archived_rows
            if isinstance(row, Mapping)
        )
        expected_observations = len(rows) * len(shard.target_shard)
        raw_integrity = bool(
            recovery["runtime_raw_sidecar_count"] >= expected_observations
            or archival_ref_count >= expected_observations
        )
        if not raw_integrity:
            raise RuntimeError("formal shard is missing raw-result sidecar evidence")
        return {
            "schema_version": "next-architecture-formal-run-loop-shard-v1",
            "corpus_digest": str(self.corpus["corpus_digest"]),
            "candidate_reservoir_digest": stable_digest(
                "formal-candidate-reservoir",
                [case.to_dict() for case in frozen_cases],
            ),
            "method_id": method.method_id,
            "method_digest": method.digest,
            "shard": shard.to_dict(),
            "config": config.to_dict(),
            "run_file": _relative_path(run_file, self.artifact_root),
            "run_meta_file": _relative_path(run_meta_path(run_file), self.artifact_root),
            "event_evidence_dir": (
                _relative_path(event_evidence_dir(run_file), self.artifact_root)
                if config.enable_event_evidence
                else ""
            ),
            "raw_archive_root": _relative_path(self.archive_root, self.artifact_root),
            "archival_raw_sidecar_count": archival_ref_count,
            "raw_evidence_integrity": raw_integrity,
            "requested_seeds": requested_seeds,
            "selected_case_ids": selected_case_ids,
            "case_rows": archived_rows,
            "coordinator": meta.get("coordinator", {}),
            "metrics": metrics,
            "replay_command": self.replay_command,
        }


def run_formal_ablation_suite(
    root: str | Path,
    *,
    root_seed: int = 8_700_000,
    block_count: int = 3,
    cases_per_block: int = 4,
    scheduler_candidate_pool: int = 3,
    targets: Sequence[str] = ("pandas", "duckdb", "buggy_filter"),
    replicates: int = 3,
    factors: Sequence[str] = FORMAL_ABLATION_FACTORS,
    max_retries: int = 1,
    factorial: str = "auto",
    replay_command: str = "",
) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selected_factors = tuple(dict.fromkeys(str(item) for item in factors if str(item)))
    unknown = sorted(set(selected_factors) - set(FORMAL_ABLATION_FACTORS))
    if unknown:
        raise ValueError(f"unknown formal ablation factors: {unknown!r}")
    if factorial not in {"auto", "never"}:
        raise ValueError("factorial must be 'auto' or 'never'")
    layout = build_formal_corpus(
        root_seed=root_seed,
        block_count=block_count,
        cases_per_block=cases_per_block,
        scheduler_candidate_pool=scheduler_candidate_pool,
    )
    environment = collect_environment()
    suite_manifest = {
        "schema_version": FORMAL_ABLATION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "parameters": {
            "root_seed": int(root_seed),
            "block_count": int(block_count),
            "cases_per_block": int(cases_per_block),
            "scheduler_candidate_pool": int(scheduler_candidate_pool),
            "targets": [str(item) for item in targets],
            "replicates": max(3, int(replicates)),
            "factors": list(selected_factors),
            "max_retries": max(0, int(max_retries)),
            "factorial": factorial,
        },
        "corpus_digest": str(layout.corpus["corpus_digest"]),
        "environment": environment,
        "replay_command": replay_command,
    }
    suite_manifest["suite_digest"] = stable_digest(
        "formal-ablation-suite",
        {key: value for key, value in suite_manifest.items() if key != "created_at"},
    )
    _initialize_immutable(root / "suite_manifest.json", suite_manifest, ignore_keys=("created_at",))
    _initialize_immutable(root / "frozen_corpus.json", layout.corpus)
    plans = build_formal_ablation_plans(layout, targets=targets, replicates=replicates)
    analyses: dict[str, dict[str, Any]] = {}
    for factor in selected_factors:
        factor_root = root / "factors" / factor
        plan = plans[factor]
        factor_root.mkdir(parents=True, exist_ok=True)
        _initialize_immutable(factor_root / "experiment_plan.json", plan.to_dict())
        _initialize_immutable(
            factor_root / "preregistration.json",
            {
                "schema_version": "next-architecture-ablation-preregistration-v1",
                "factor": factor,
                "plan_digest": plan.digest,
                "gate_rules": [
                    {"metric": metric, "expected_direction": direction}
                    for metric, direction in _GATE_RULES[factor]
                ],
                "replay_command": replay_command,
                "claim_boundary": (
                    "seeded fault detections measure sensitivity/cost only and are not independent discovery claims"
                ),
            },
        )
        analyses[factor] = CampaignRunner(
            factor_root,
            plan,
            environment=environment,
            execute_shard=FrozenRunLoopShardExecutor(
                layout.corpus,
                artifact_root=factor_root,
                replay_command=replay_command,
            ),
            max_retries=max_retries,
            workers=1,
        ).run()
    stability = build_stability_gate(analyses)
    dump_json(stability, root / "stability_gate.json")
    factorial_analysis: dict[str, Any] | None = None
    if factorial == "auto" and stability["factorial_eligible"]:
        factorial_plan = build_factorial_plan(
            layout,
            targets=tuple(str(item) for item in targets if str(item)),
            replicates=max(3, int(replicates)),
        )
        factorial_root = root / "factorial"
        factorial_root.mkdir(parents=True, exist_ok=True)
        _initialize_immutable(factorial_root / "experiment_plan.json", factorial_plan.to_dict())
        factorial_analysis = CampaignRunner(
            factorial_root,
            factorial_plan,
            environment=environment,
            execute_shard=FrozenRunLoopShardExecutor(
                layout.corpus,
                artifact_root=factorial_root,
                replay_command=replay_command,
            ),
            max_retries=max_retries,
            workers=1,
        ).run()
    integrity = audit_formal_ablation_suite(root, expected_factors=selected_factors)
    result = {
        "schema_version": FORMAL_ABLATION_SCHEMA_VERSION,
        "suite_digest": suite_manifest["suite_digest"],
        "corpus_digest": str(layout.corpus["corpus_digest"]),
        "factors": analyses,
        "stability_gate": stability,
        "factorial": {
            "requested": factorial,
            "ran": factorial_analysis is not None,
            "analysis": factorial_analysis or {},
            "blocked_reason": (
                ""
                if factorial_analysis is not None or factorial == "never"
                else "one or more preregistered single-factor effects were not stable"
            ),
        },
        "integrity_audit": integrity,
    }
    dump_json(result, root / "analysis.json")
    (root / "report.md").write_text(_suite_report(result), encoding="utf-8")
    return result


def build_stability_gate(analyses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    factors: dict[str, Any] = {}
    for factor in FORMAL_ABLATION_FACTORS:
        analysis = analyses.get(factor)
        judgments: list[dict[str, Any]] = []
        for metric, direction in _GATE_RULES[factor]:
            effect = _paired_effect(analysis or {}, metric)
            interval = effect.get("bootstrap_95_ci", [0.0, 0.0])
            low = float(interval[0] if len(interval) > 0 else 0.0)
            high = float(interval[1] if len(interval) > 1 else low)
            stable = bool(
                int(effect.get("paired_block_count", 0) or 0) >= 9
                and ((high < 0.0) if direction == "decrease" else (low > 0.0))
            )
            judgments.append(
                {
                    "metric": metric,
                    "expected_direction": direction,
                    "paired_block_count": int(effect.get("paired_block_count", 0) or 0),
                    "mean_delta": float(effect.get("mean_delta", 0.0) or 0.0),
                    "bootstrap_95_ci": [low, high],
                    "stable": stable,
                }
            )
        factors[factor] = {
            "present": analysis is not None,
            "stable": bool(judgments and all(item["stable"] for item in judgments)),
            "judgments": judgments,
        }
    return {
        "schema_version": "next-architecture-stability-gate-v1",
        "factors": factors,
        "factorial_eligible": all(
            factors[factor]["present"] and factors[factor]["stable"]
            for factor in FORMAL_ABLATION_FACTORS
        ),
        "policy": (
            "all five single-factor plans must satisfy their preregistered paired-bootstrap direction gates"
        ),
    }


def build_factorial_plan(
    layout: FormalCorpusLayout,
    *,
    targets: tuple[str, ...],
    replicates: int,
) -> ExperimentPlan:
    base = replace(_base_method("scheduler", layout, targets=targets), method_id="factorial_000")
    methods = [base]
    for evidence_on in (0, 1):
        for parallel_on in (0, 1):
            for scheduler_on in (0, 1):
                if not (evidence_on or parallel_on or scheduler_on):
                    continue
                changed: list[str] = []
                scheduling = dict(base.scheduling)
                execution = dict(base.execution)
                evidence = dict(base.evidence)
                if scheduler_on:
                    changed.append("scheduling")
                    scheduling.update(
                        {
                            "scheduler": "coverage-debt-v1",
                            "coverage_debt": True,
                            "fuzz_config_overrides": {
                                "enable_coordinator_coverage_debt": True,
                            },
                        }
                    )
                if parallel_on:
                    changed.append("execution")
                    execution.update(
                        {
                            "topology": "same-case-all-backends-parallel",
                            "fuzz_config_overrides": {
                                "enable_parallel_backend_execution": True,
                            },
                        }
                    )
                if evidence_on:
                    changed.append("evidence")
                    evidence.update(
                        {
                            "plan_collection": "tiered",
                            "fuzz_config_overrides": {
                                "method_arm_overrides": {
                                    **_COMMON_METHOD_OVERRIDES,
                                    "plan_collection_mode": "tiered",
                                }
                            },
                        }
                    )
                methods.append(
                    replace(
                        base,
                        method_id=f"factorial_{evidence_on}{parallel_on}{scheduler_on}",
                        parent_method_id=base.method_id,
                        changed_dimensions=tuple(changed),
                        scheduling=scheduling,
                        execution=execution,
                        evidence=evidence,
                    )
                )
    return ExperimentPlan(
        plan_id="next-architecture-evidence-parallel-scheduler-factorial-v1",
        design="factorial",
        methods=tuple(methods),
        seed_blocks=layout.scheduler_blocks,
        target_shards=(targets,),
        replicates=max(3, int(replicates)),
        corpus_manifest_digest=str(layout.corpus["corpus_digest"]),
        resource_budget={
            "executed_cases_per_shard": layout.cases_per_block,
            "candidate_pool": layout.scheduler_candidate_pool,
            "workers": 1,
        },
        stopping_rule={
            "kind": "complete_declared_matrix",
            "requires_stable_single_factor_gate": True,
        },
        primary_metrics=(
            "independent_roots",
            "coverage_cells",
            "wall_ms",
            "cpu_ms",
            "rss_mib",
            "io_bytes",
        ),
        secondary_metrics=("candidate_count", "evidence_completeness_rate"),
        exclusion_rules=(
            "factorial execution is forbidden unless stability_gate.factorial_eligible is true",
        ),
    )


def audit_formal_ablation_suite(
    root: str | Path,
    *,
    expected_factors: Sequence[str] = FORMAL_ABLATION_FACTORS,
) -> dict[str, Any]:
    root = Path(root)
    suite_manifest = load_json(root / "suite_manifest.json")
    corpus = load_json(root / "frozen_corpus.json")
    validate_frozen_case_corpus(corpus)
    factor_results: dict[str, Any] = {}
    for factor in expected_factors:
        factor_root = root / "factors" / factor
        manifest_path = factor_root / "experiment_manifest.json"
        analysis_path = factor_root / "analysis.json"
        shard_path = factor_root / "shard_manifest.jsonl"
        metric_path = factor_root / "metric_rows.jsonl"
        errors: list[str] = []
        if not all(path.exists() for path in (manifest_path, analysis_path, shard_path, metric_path)):
            factor_results[factor] = {"pass": False, "errors": ["missing campaign artifact"]}
            continue
        manifest = load_json(manifest_path)
        analysis = load_json(analysis_path)
        plan = manifest.get("plan", {}) if isinstance(manifest.get("plan", {}), Mapping) else {}
        environment = manifest.get("environment", {})
        if str(plan.get("corpus_manifest_digest", "")) != str(corpus["corpus_digest"]):
            errors.append("corpus digest mismatch")
        if not str((environment or {}).get("source_tree_sha256", "")):
            errors.append("missing source tree digest")
        shard_records: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(shard_path):
            if row.get("shard_id"):
                shard_records[str(row["shard_id"])] = row
        completed = [row for row in shard_records.values() if row.get("status") == "completed"]
        expected = int((analysis.get("completion", {}) or {}).get("expected_shards", 0) or 0)
        if len(completed) != expected or not bool((analysis.get("completion", {}) or {}).get("complete")):
            errors.append("incomplete shard matrix")
        metric_rows = read_jsonl(metric_path)
        if len(metric_rows) != expected:
            errors.append("metric row count mismatch")
        raw_integrity = True
        replay_commands: set[str] = set()
        for row in completed:
            output_ref = str(row.get("output_ref", "") or "")
            output_path = factor_root / output_ref
            if not output_path.is_file():
                errors.append(f"missing shard output: {output_ref}")
                raw_integrity = False
                continue
            output = load_json(output_path)
            raw_integrity = raw_integrity and bool(output.get("raw_evidence_integrity"))
            command = str(output.get("replay_command", "") or "")
            if command:
                replay_commands.add(command)
            if not isinstance(row.get("retry_history", []), list):
                errors.append("retry history is not recorded")
        if not raw_integrity:
            errors.append("raw-result sidecar integrity failed")
        if not replay_commands and not str(suite_manifest.get("replay_command", "") or ""):
            errors.append("missing replay command")
        factor_results[factor] = {
            "pass": not errors,
            "errors": errors,
            "expected_shards": expected,
            "completed_shards": len(completed),
            "metric_rows": len(metric_rows),
            "failed_shards": [
                row.get("shard_id")
                for row in shard_records.values()
                if row.get("status") == "failed"
            ],
            "replay_commands": sorted(replay_commands),
        }
    result = {
        "schema_version": "next-architecture-formal-integrity-audit-v1",
        "source_tree_sha256": str(
            (suite_manifest.get("environment", {}) or {}).get("source_tree_sha256", "")
        ),
        "corpus_digest": str(corpus["corpus_digest"]),
        "factors": factor_results,
        "all_pass": bool(factor_results)
        and all(item.get("pass") for item in factor_results.values()),
    }
    dump_json(result, root / "integrity_audit.json")
    (root / "integrity_audit.md").write_text(_integrity_report(result), encoding="utf-8")
    return result


def _runtime_artifact_metrics(run_file: Path) -> dict[str, float]:
    evidence_dir = event_evidence_dir(run_file)
    run_log_bytes = float(run_file.stat().st_size if run_file.is_file() else 0)
    meta_path = run_meta_path(run_file)
    meta_bytes = float(meta_path.stat().st_size if meta_path.is_file() else 0)
    sidecar_bytes = float(_path_bytes(evidence_dir))
    return {
        "run_log_bytes": run_log_bytes,
        "run_meta_bytes": meta_bytes,
        "sidecar_bytes": sidecar_bytes,
        "artifact_bytes": run_log_bytes + meta_bytes + sidecar_bytes,
    }


def _recovery_metrics(
    run_file: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    event_enabled: bool,
) -> dict[str, float]:
    if not event_enabled:
        return {
            "evidence_completeness_rate": 0.0,
            "recovery_success_rate": 0.0,
            "runtime_raw_sidecar_count": 0.0,
        }
    evidence_dir = event_evidence_dir(run_file)
    manifest_ok = (evidence_dir / "run_manifest.json").is_file()
    events = read_jsonl(evidence_dir / "events.jsonl")
    case_digests = {
        str(event.get("payload", {}).get("manifest_digest", "") or "")
        for event in events
        if event.get("event_type") == "case_manifest"
    }
    referenced_case_digests = {
        str((row.get("evidence_refs", {}) or {}).get("case_manifest_digest", "") or "")
        for row in rows
    }
    raw_sidecars = 0
    raw_sidecars_ok = True
    for event in events:
        if event.get("event_type") != "observation":
            continue
        reference = event.get("payload", {}).get("raw_result_ref")
        if not isinstance(reference, Mapping):
            raw_sidecars_ok = False
            continue
        raw_sidecars += 1
        path = evidence_dir / str(reference.get("relative_path", "") or "")
        if not path.is_file() or int(reference.get("byte_count", -1) or -1) != path.stat().st_size:
            raw_sidecars_ok = False
    complete = bool(
        manifest_ok
        and len(case_digests) == len(rows)
        and referenced_case_digests == case_digests
        and raw_sidecars_ok
    )
    rate = 1.0 if complete else 0.0
    return {
        "evidence_completeness_rate": rate,
        "recovery_success_rate": rate,
        "runtime_raw_sidecar_count": float(raw_sidecars),
    }


def _analysis_elapsed_ms(run_file: Path, *, include_events: bool) -> float:
    started = time.perf_counter()
    for _ in range(3):
        read_jsonl(run_file)
        if include_events:
            read_jsonl(event_evidence_dir(run_file) / "events.jsonl")
    return max(0.0, (time.perf_counter() - started) * 1000.0 / 3.0)


def _row_max_rss(rows: Sequence[Mapping[str, Any]]) -> float:
    maximum = 0.0
    for row in rows:
        raw_results = row.get("raw_results", {})
        if not isinstance(raw_results, Mapping):
            continue
        for raw in raw_results.values():
            if isinstance(raw, Mapping):
                maximum = max(maximum, _number(raw.get("rss_mib", 0.0)))
    return maximum


def _plan_evidence_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    plan_observations = 0
    full_plans = 0
    fingerprints = 0
    for row in rows:
        raw_results = row.get("raw_results", {})
        if not isinstance(raw_results, Mapping):
            continue
        for raw in raw_results.values():
            if not isinstance(raw, Mapping):
                continue
            plan = raw.get("physical_plan")
            if not isinstance(plan, Mapping):
                continue
            plan_observations += 1
            detail = str(plan.get("detail", "") or "")
            full_plans += int(detail == "full")
            fingerprints += int(detail == "fingerprint")
    return {
        "plan_observation_count": float(plan_observations),
        "full_plan_evidence_rate": (
            full_plans / plan_observations if plan_observations else 0.0
        ),
        "plan_fingerprint_rate": (
            fingerprints / plan_observations if plan_observations else 0.0
        ),
    }


def _paired_effect(analysis: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    for effect in analysis.get("paired_effects", ()) or ():
        if isinstance(effect, Mapping) and str(effect.get("metric", "")) == metric:
            return effect
    return {}


def _nested_number(payload: Mapping[str, Any], *path: str) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return 0.0
        value = value.get(key)
    return _number(value)


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _initialize_immutable(
    path: Path,
    payload: Mapping[str, Any],
    *,
    ignore_keys: Sequence[str] = (),
) -> None:
    if path.exists():
        existing = load_json(path)
        comparable_existing = {key: value for key, value in existing.items() if key not in ignore_keys}
        comparable_payload = {key: value for key, value in payload.items() if key not in ignore_keys}
        if comparable_existing != comparable_payload:
            raise ValueError(f"immutable formal artifact differs: {path}")
        return
    dump_json(dict(payload), path)


def _suite_report(result: Mapping[str, Any]) -> str:
    gate = result.get("stability_gate", {})
    lines = [
        "# Next architecture formal ablations",
        "",
        f"- Corpus digest: `{result.get('corpus_digest', '')}`",
        f"- Integrity audit passed: `{result.get('integrity_audit', {}).get('all_pass', False)}`",
        f"- Factorial eligible: `{gate.get('factorial_eligible', False)}`",
        f"- Factorial ran: `{result.get('factorial', {}).get('ran', False)}`",
        "",
        "## Stability gate",
        "",
    ]
    for factor, item in (gate.get("factors", {}) or {}).items():
        lines.append(f"- {factor}: stable=`{item.get('stable', False)}`")
        for judgment in item.get("judgments", ()):
            interval = judgment.get("bootstrap_95_ci", [0.0, 0.0]) or [0.0, 0.0]
            lines.append(
                "  - {metric}: delta={delta:.6f}, CI=[{low:.6f}, {high:.6f}], expected={direction}".format(
                    metric=judgment.get("metric", ""),
                    delta=float(judgment.get("mean_delta", 0.0) or 0.0),
                    low=float(interval[0]),
                    high=float(interval[-1]),
                    direction=judgment.get("expected_direction", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Seeded faulty targets are used only for sensitivity and grouping signal. They do not count as independent upstream roots.",
            "P7/RLCMF remains shadow and P8 remains candidate; this suite performs no promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def _integrity_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Formal ablation integrity audit",
        "",
        f"- All factors passed: `{result.get('all_pass', False)}`",
        f"- Source tree SHA-256: `{result.get('source_tree_sha256', '')}`",
        f"- Corpus digest: `{result.get('corpus_digest', '')}`",
        "",
    ]
    for factor, item in (result.get("factors", {}) or {}).items():
        lines.append(
            f"- {factor}: pass=`{item.get('pass', False)}`, completed={item.get('completed_shards', 0)}/{item.get('expected_shards', 0)}, metrics={item.get('metric_rows', 0)}"
        )
        for error in item.get("errors", ()):
            lines.append(f"  - error: {error}")
    lines.append("")
    return "\n".join(lines)
