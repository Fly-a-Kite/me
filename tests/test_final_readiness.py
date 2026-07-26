import os
from pathlib import Path

from datadiff import final_readiness
from datadiff.experiment_catalog import (
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX,
    FINAL_COMPARISON_MATRIX,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    FINAL_VALIDATION_MATRIX,
)
from datadiff.final_readiness import (
    DEFAULT_A_LEVEL_READINESS_POLICY,
    DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
    ReadinessPolicy,
    ReadinessThresholds,
    build_final_readiness,
)
from datadiff.targets import TARGET_SUITES, describe_targets
from datadiff.util import append_jsonl, dump_json, load_json, run_meta_path
from datadiff.version_ledger import VersionObservation, build_version_ledger


REQUIRED_ADAPTIVE_COMPONENTS = (
    "active_learning",
    "backend_pair_learning",
    "bd_axis_bandit",
    "bayesian_exploration",
    "champion_graft_donor",
    "continual_learning",
    "cost_normalized_reward",
    "disagreement_bd_axis",
    "divergence_conditioned",
    "hierarchical_archive",
    "ir_rewrite_mutations",
    "lineage_rarity",
    "minhash_dedup",
    "online_reward_model",
    "operator_swarm",
    "value_catalog",
    "quality_archive",
    "seed_quota",
    "seed_energy_batch",
    "seed_energy_tier",
    "per_operator_energy",
    "lhs_seeding",
    "champion_corpus",
    "runtime_cost_learning",
    "scheduler_annealing",
    "scheduler_learning",
    "shrink_mutations",
)


def _adaptive_component_flags(**overrides: bool) -> dict[str, bool]:
    flags = {component: True for component in REQUIRED_ADAPTIVE_COMPONENTS}
    flags.update(overrides)
    return flags


def _adaptive_component_ablation_manifest(
    root: Path,
    *,
    component: str,
    seed: int,
) -> Path:
    return _write_manifest(
        root,
        name=f"ablation-adaptive-{component.replace('_', '-')}",
        evidence_mode="ablation",
        target_suite="datafusion_cross",
        preset="live_deep_organic",
        seed=seed,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={
            "adaptive_components": _adaptive_component_flags(
                **{component: False, "local_source_scheduler": True},
            ),
            "disabled_adaptive_components": [component],
        },
        experiment_meta={
            **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                target_suites=("datafusion_cross",)
            ),
            "variant": {
                "variant_id": f"no_{component}",
                "comparison_role": "contrast",
                "component_focus": component,
                "factors": {component: False},
            },
        },
    )


def test_final_readiness_passes_when_all_evidence_tracks_are_present(tmp_path):
    manifests = []
    version_ledger_path = _write_version_ledger(tmp_path)
    target_version_audit_path = _write_target_version_audit(tmp_path, outdated=False)
    manifests.append(target_version_audit_path)
    manifests.append(
        _write_manifest(
            tmp_path,
            name="validation",
            evidence_mode="validation",
            target_suite="datafusion_cross",
            preset="validation_smoke",
            seed=0,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_VALIDATION_MATRIX.command_experiment_meta(
                target_suites=("datafusion_cross",),
                preset=FINAL_VALIDATION_MATRIX.variants[0].preset,
            ),
        )
    )
    for idx, suite in enumerate(DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites, start=1):
        finding = []
        if suite == "datafusion_cross":
            finding = [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "confirmed_root",
                    "suspicious_backends": ["datafusion"],
                    "discovery_origin": "organic",
                    "paper_status": "confirmed_bug",
                }
            ]
        manifests.append(
            _write_manifest(
                tmp_path,
                name=f"live-{suite}",
                evidence_mode="live",
                target_suite=suite,
                preset="live",
                seed=idx,
                findings=finding,
                config={"enable_replay_bug": False},
                replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
                first_candidate_elapsed_s=0.25 if finding else None,
                first_candidate_case_index=0 if finding else None,
                candidate_bug_discovery_auc=1.0 if finding else 0.0,
                experiment_meta=FINAL_LIVE_DISCOVERY_MATRIX.command_experiment_meta(
                    target_suites=(suite,),
                    preset=FINAL_LIVE_DISCOVERY_MATRIX.variants[0].preset,
                    counts_as_real_bugs=True,
                ),
            )
        )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="historical-duckdb-22075",
            evidence_mode="historical",
            target_suite="cross_family",
            preset="join_groupby_stress",
            seed=22075,
            known_bug_id="duckdb-22075",
            config={"enable_replay_bug": True},
            experiment_meta={
                "matrix_id": "historical_replay",
                "comparison_group": "historical_replay",
                "variant": {"variant_id": "duckdb-22075", "comparison_role": "contrast"},
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="historical-duckdb-22656",
            evidence_mode="historical",
            target_suite="duckdb_storage_cross",
            preset="storage_offset",
            seed=22656,
            known_bug_id="duckdb-22656",
            config={"enable_replay_bug": True},
            experiment_meta={
                "matrix_id": "historical_replay",
                "comparison_group": "historical_replay",
                "variant": {"variant_id": "duckdb-22656", "comparison_role": "contrast"},
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="seeded",
            evidence_mode="seeded",
            target_suite="seeded_filter",
            preset="guided_filter",
            seed=1,
            experiment_meta=FINAL_SEEDED_SENSITIVITY_MATRIX.command_experiment_meta(
                target_suites=("seeded_filter",),
                preset="guided_filter",
            ),
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-module-baseline",
            evidence_mode="ablation",
            target_suite="core",
            preset="baseline",
            seed=0,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
                target_suites=("core",),
                preset="baseline",
            ),
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-quality-archive",
            evidence_mode="ablation",
            target_suite="core",
            preset="no_normalizer",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": {
                    "scheduler_learning": True,
                    "quality_archive": False,
                    "local_source_scheduler": True,
                },
                "disabled_adaptive_components": ["quality_archive"],
            },
            experiment_meta=FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
                target_suites=("core",),
                preset="no_normalizer",
            ),
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-runtime-cost",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=2,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    runtime_cost_learning=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["runtime_cost_learning"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_runtime_cost_learning",
                    "comparison_role": "contrast",
                    "component_focus": "runtime_cost_learning",
                    "factors": {"runtime_cost_learning": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-reference",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=3,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(local_source_scheduler=True),
                "disabled_adaptive_components": [],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "adaptive_reference",
                    "comparison_role": "baseline",
                    "component_focus": "adaptive_closed_loop",
                    "factors": {
                        "adaptive_closed_loop": True,
                        "scheduler_learning": True,
                        "online_reward_model": True,
                        "continual_learning": True,
                        "active_learning": True,
                        "bayesian_exploration": True,
                        "bd_axis_bandit": True,
                        "value_catalog": True,
                        "quality_archive": True,
                        "seed_quota": True,
                        "seed_energy_batch": True,
                        "per_operator_energy": True,
                        "lhs_seeding": True,
                        "champion_corpus": True,
                        "runtime_cost_learning": True,
                        "scheduler_annealing": True,
                    },
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-scheduler-learning",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=6,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    scheduler_learning=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["scheduler_learning"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_scheduler_learning",
                    "comparison_role": "contrast",
                    "component_focus": "scheduler_learning",
                    "factors": {"scheduler_learning": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-scheduler-annealing",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=9,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    scheduler_annealing=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["scheduler_annealing"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_scheduler_annealing",
                    "comparison_role": "contrast",
                    "component_focus": "scheduler_annealing",
                    "factors": {"scheduler_annealing": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-online-reward-model",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=7,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    online_reward_model=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["online_reward_model"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_online_reward_model",
                    "comparison_role": "contrast",
                    "component_focus": "online_reward_model",
                    "factors": {"online_reward_model": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-continual-learning",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=8,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    continual_learning=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["continual_learning"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_continual_learning",
                    "comparison_role": "contrast",
                    "component_focus": "continual_learning",
                    "factors": {"continual_learning": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-quality-archive",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=4,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    quality_archive=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["quality_archive"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_quality_archive",
                    "comparison_role": "contrast",
                    "component_focus": "quality_archive",
                    "factors": {"quality_archive": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-bd-axis-bandit",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=18,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    bd_axis_bandit=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["bd_axis_bandit"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_bd_axis_bandit",
                    "comparison_role": "contrast",
                    "component_focus": "bd_axis_bandit",
                    "factors": {"bd_axis_bandit": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-bayesian-exploration",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=19,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    bayesian_exploration=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["bayesian_exploration"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_bayesian_exploration",
                    "comparison_role": "contrast",
                    "component_focus": "bayesian_exploration",
                    "factors": {"bayesian_exploration": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-value-catalog",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=14,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    value_catalog=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["value_catalog"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_value_catalog",
                    "comparison_role": "contrast",
                    "component_focus": "value_catalog",
                    "factors": {"value_catalog": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-seed-quota",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=11,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    seed_quota=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["seed_quota"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_seed_quota",
                    "comparison_role": "contrast",
                    "component_focus": "seed_quota",
                    "factors": {"seed_quota": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-seed-energy-batch",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=15,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    seed_energy_batch=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["seed_energy_batch"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_seed_energy_batch",
                    "comparison_role": "contrast",
                    "component_focus": "seed_energy_batch",
                    "factors": {"seed_energy_batch": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-per-operator-energy",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=16,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    per_operator_energy=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["per_operator_energy"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_per_operator_energy",
                    "comparison_role": "contrast",
                    "component_focus": "per_operator_energy",
                    "factors": {"per_operator_energy": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-ir-rewrite-mutations",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=17,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    ir_rewrite_mutations=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["ir_rewrite_mutations"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_ir_rewrite_mutations",
                    "comparison_role": "contrast",
                    "component_focus": "ir_rewrite_mutations",
                    "factors": {"ir_rewrite_mutations": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-lhs-seeding",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=12,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    lhs_seeding=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["lhs_seeding"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_lhs_seeding",
                    "comparison_role": "contrast",
                    "component_focus": "lhs_seeding",
                    "factors": {"lhs_seeding": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-champion-corpus",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=13,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    champion_corpus=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["champion_corpus"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_champion_corpus",
                    "comparison_role": "contrast",
                    "component_focus": "champion_corpus",
                    "factors": {"champion_corpus": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-active-learning",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=5,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    active_learning=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["active_learning"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_active_learning",
                    "comparison_role": "contrast",
                    "component_focus": "active_learning",
                    "factors": {"active_learning": False},
                },
            },
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="comparison",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={"version_ledger_file": str(version_ledger_path)},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="baseline",
            ),
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-backend-pair-learning",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=10,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(
                    backend_pair_learning=False,
                    local_source_scheduler=True,
                ),
                "disabled_adaptive_components": ["backend_pair_learning"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_backend_pair_learning",
                    "comparison_role": "contrast",
                    "component_focus": "backend_pair_learning",
                    "factors": {"backend_pair_learning": False},
                },
            },
        )
    )
    for offset, component in enumerate(
        (
            "operator_swarm",
            "divergence_conditioned",
            "shrink_mutations",
            "hierarchical_archive",
            "lineage_rarity",
            "minhash_dedup",
            "disagreement_bd_axis",
            "cost_normalized_reward",
            "seed_energy_tier",
            "champion_graft_donor",
        ),
        start=20,
    ):
        manifests.append(
            _adaptive_component_ablation_manifest(
                tmp_path,
                component=component,
                seed=offset,
            )
        )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="comparison-guided",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="guided",
            seed=2,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={"version_ledger_file": str(version_ledger_path)},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="guided",
            ),
        )
    )

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(min_live_duration_hours=0.0),
    )

    assert audit["schema_version"] == "final-readiness-v1"
    assert audit["ready"] is True
    assert audit["icse_experiment_quality"]["schema_version"] == "icse-experiment-quality-v1"
    assert audit["summary"]["icse_experiment_quality"] == audit["icse_experiment_quality"]
    target_version_gate = {gate["name"]: gate for gate in audit["gates"]}["target_version_audit"]
    assert target_version_gate["passed"] is True
    assert audit["summary"]["target_version_audit"]["valid_audit_files"] == [str(target_version_audit_path)]
    assert set(audit["icse_experiment_quality"]["dimensions"]) == {
        "real_bug_yield",
        "throughput",
        "coverage",
        "speed",
        "reproducibility",
    }
    assert tuple(audit["policy"]["required_live_suites"]) == DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["short_validation"] is True
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is True
    assert audit["summary"]["validation_runs"] == 1
    assert audit["summary"]["validation_suites"] == ["datafusion_cross"]
    assert audit["summary"]["ablation_runs"] == 2
    assert audit["summary"]["module_ablation_comparison"]["complete_group_count"] == 1
    assert audit["summary"]["module_ablation_comparison"]["reference_run_count"] == 1
    assert audit["summary"]["module_ablation_comparison"]["contrast_run_count"] == 1
    assert audit["summary"]["adaptive_component_ablation"]["reference_run_count"] == 1
    assert audit["summary"]["adaptive_component_ablation"]["contrast_run_count"] == len(
        REQUIRED_ADAPTIVE_COMPONENTS
    )
    assert audit["summary"]["adaptive_component_ablation"]["disabled_components"] == sorted(
        REQUIRED_ADAPTIVE_COMPONENTS
    )
    assert audit["summary"]["adaptive_component_ablation"]["reference_metrics"]["cases"] == 1
    assert audit["summary"]["adaptive_component_ablation"]["contrast_metrics"]["cases"] == len(
        REQUIRED_ADAPTIVE_COMPONENTS
    )
    assert audit["summary"]["final_matrix_coverage"]["missing_matrix_ids"] == []
    assert audit["summary"]["cross_version_ledger"]["ledger_count"] == 1
    assert audit["summary"]["cross_version_ledger"]["max_version_count"] == 2
    assert audit["summary"]["cross_version_ledger"]["transition_counts"]["new"] == 1
    assert audit["summary"]["cross_version_ledger"]["champion_transfer"]["ledger_count"] == 1
    assert audit["summary"]["discovery_responsiveness"]["best_first_candidate_elapsed_s"] == 0.25
    assert audit["summary"]["comparison_runs"] == 2
    assert audit["summary"]["baseline_comparison"]["complete_group_count"] == 1
    assert audit["summary"]["baseline_comparison"]["reference_run_count"] == 1
    assert audit["summary"]["baseline_comparison"]["contrast_run_count"] == 1
    assert audit["summary"]["paper_run_journal_covered_runs"] == len(audit["runs"])
    assert audit["summary"]["paper_run_journal_missing_runs"] == []
    assert audit["summary"]["confirmed_live_candidate_families"] == {"confirmed_root@datafusion": 1}
    assert audit["summary"]["historical_confirmed_bug_ids"] == ["duckdb-22075", "duckdb-22656"]


def test_final_readiness_requires_adaptive_component_ablation_for_final_claim(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="ablation-without-adaptive-components",
        evidence_mode="ablation",
        target_suite="core",
        preset="no_normalizer",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["adaptive_component_ablation"]["passed"] is False
    assert gates["adaptive_component_ablation"]["disabled_components"] == []
    assert sorted(gates["adaptive_component_ablation"]["missing_required_components"]) == sorted(
        REQUIRED_ADAPTIVE_COMPONENTS
    )


def test_final_readiness_requires_key_adaptive_component_ablations(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="ablation-adaptive-reference",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=0,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(),
                "disabled_adaptive_components": [],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "adaptive_reference",
                    "comparison_role": "baseline",
                    "component_focus": "adaptive_closed_loop",
                },
            },
        ),
        _write_manifest(
            tmp_path,
            name="ablation-only-quality-archive",
            evidence_mode="ablation",
            target_suite="datafusion_cross",
            preset="live_deep_organic",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "adaptive_components": _adaptive_component_flags(quality_archive=False),
                "disabled_adaptive_components": ["quality_archive"],
            },
            experiment_meta={
                **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                    target_suites=("datafusion_cross",)
                ),
                "variant": {
                    "variant_id": "no_quality_archive",
                    "comparison_role": "contrast",
                    "component_focus": "quality_archive",
                    "factors": {"quality_archive": False},
                },
            },
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_comparison=False,
            require_transferability_scope=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["adaptive_component_ablation"]["passed"] is False
    assert gates["adaptive_component_ablation"]["disabled_components"] == ["quality_archive"]
    assert gates["adaptive_component_ablation"]["missing_required_components"] == sorted(
        set(REQUIRED_ADAPTIVE_COMPONENTS) - {"quality_archive"}
    )
    assert gates["adaptive_component_ablation"]["reference_run_count"] == 1
    assert gates["adaptive_component_ablation"]["contrast_run_count"] == 1


def test_final_readiness_requires_adaptive_component_reference_run(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="ablation-without-adaptive-reference",
        evidence_mode="ablation",
        target_suite="datafusion_cross",
        preset="live_deep_organic",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={
            "adaptive_components": _adaptive_component_flags(
                active_learning=False,
                bayesian_exploration=False,
                backend_pair_learning=False,
                bd_axis_bandit=False,
                champion_corpus=False,
                continual_learning=False,
                ir_rewrite_mutations=False,
                lhs_seeding=False,
                online_reward_model=False,
                per_operator_energy=False,
                quality_archive=False,
                runtime_cost_learning=False,
                scheduler_annealing=False,
                scheduler_learning=False,
                seed_energy_batch=False,
                seed_quota=False,
                value_catalog=False,
            ),
            "disabled_adaptive_components": list(REQUIRED_ADAPTIVE_COMPONENTS),
        },
        experiment_meta={
            **FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
                target_suites=("datafusion_cross",)
            ),
            "variant": {
                "variant_id": "no_quality_archive_runtime_cost",
                "comparison_role": "contrast",
                "component_focus": "adaptive_closed_loop",
                "factors": {
                    "active_learning": False,
                    "backend_pair_learning": False,
                    "bd_axis_bandit": False,
                    "champion_corpus": False,
                    "continual_learning": False,
                    "ir_rewrite_mutations": False,
                    "lhs_seeding": False,
                    "online_reward_model": False,
                    "per_operator_energy": False,
                    "quality_archive": False,
                    "runtime_cost_learning": False,
                    "scheduler_annealing": False,
                    "scheduler_learning": False,
                    "seed_energy_batch": False,
                    "seed_quota": False,
                    "value_catalog": False,
                },
            },
        },
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_comparison=False,
            require_transferability_scope=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_component_ablation"]
    assert gate["passed"] is False
    assert gate["missing_required_components"] == []
    assert gate["reference_run_count"] == 0
    assert gate["contrast_run_count"] == 1


def test_final_readiness_requires_standard_final_matrix_coverage(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-only-matrix",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        experiment_meta=FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=("datafusion_cross",),
            preset=FINAL_VALIDATION_MATRIX.variants[0].preset,
        ),
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_transferability_scope=False,
            require_cross_version_ledger=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["final_matrix_coverage"]["passed"] is False
    assert gates["final_matrix_coverage"]["observed_matrix_ids"] == ["final_validation"]
    assert "adaptive_component_ablation" in gates["final_matrix_coverage"]["missing"]
    assert "live_discovery" in gates["final_matrix_coverage"]["missing"]


def test_final_readiness_requires_cross_version_ledger_for_continual_claim(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-without-ledger",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        experiment_meta=FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=("datafusion_cross",),
            preset=FINAL_VALIDATION_MATRIX.variants[0].preset,
        ),
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["cross_version_continual_learning"]
    assert gate["passed"] is False
    assert gate["ledger_files"] == []


def test_final_readiness_rejects_invalid_cross_version_ledger(tmp_path):
    invalid_ledger = tmp_path / "reports" / "invalid-ledger.json"
    dump_json({"schema_version": "not-a-version-ledger"}, invalid_ledger)
    manifest = _write_manifest(
        tmp_path,
        name="validation-invalid-ledger",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={"version_ledger_file": str(invalid_ledger)},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["cross_version_continual_learning"]
    assert gate["passed"] is False
    assert gate["invalid_reasons"][str(invalid_ledger)] == "schema_mismatch"


def test_final_readiness_requires_target_version_audit_for_live_claim(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-without-target-version-audit",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
            require_cross_version_ledger=False,
            require_closed_loop_state_persistence=False,
            require_adaptive_live_component_evidence=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["target_version_audit"]
    assert gate["passed"] is False
    assert gate["valid_audit_files"] == []
    assert audit["summary"]["target_version_audit"]["valid_audit_count"] == 0


def test_final_readiness_accepts_up_to_date_target_version_audit(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-with-target-version-audit",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    version_audit = _write_target_version_audit(tmp_path, outdated=False)

    audit = build_final_readiness(
        [manifest, version_audit],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
            require_cross_version_ledger=False,
            require_closed_loop_state_persistence=False,
            require_adaptive_live_component_evidence=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["target_version_audit"]
    assert gate["passed"] is True
    assert gate["valid_audit_files"] == [str(version_audit)]
    assert audit["summary"]["target_version_audit"]["all_target_packages_up_to_date"] is True


def test_final_readiness_rejects_outdated_target_version_audit(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-with-outdated-target-version-audit",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    version_audit = _write_target_version_audit(tmp_path, outdated=True)

    audit = build_final_readiness(
        [manifest, version_audit],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
            require_cross_version_ledger=False,
            require_closed_loop_state_persistence=False,
            require_adaptive_live_component_evidence=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["target_version_audit"]
    assert gate["passed"] is False
    assert gate["outdated_target_packages"][0]["package"] == "pandas"


def test_final_readiness_rejects_old_cross_version_ledger_without_health_feedback(tmp_path):
    old_ledger = tmp_path / "reports" / "old-ledger.json"
    dump_json(
        {
            "schema_version": "version-ledger-v1",
            "baseline_version": "v1",
            "version_order": ["v1", "v2"],
            "summary": {
                "version_count": 2,
                "family_count": 1,
                "transition_counts": {"new": 1},
                "new_family_count": 1,
                "fixed_family_count": 0,
                "regression_family_count": 0,
                "persistent_family_count": 0,
            },
            "families": [
                {
                    "family": "new_root@engine",
                    "status": "new",
                    "observations": [
                        {"version_id": "v1", "state": "absent", "count": 0},
                        {"version_id": "v2", "state": "present", "count": 1},
                    ],
                }
            ],
        },
        old_ledger,
    )
    manifest = _write_manifest(
        tmp_path,
        name="validation-old-ledger",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={"version_ledger_file": str(old_ledger)},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["cross_version_continual_learning"]
    summary = audit["summary"]["cross_version_ledger"]
    assert gate["passed"] is False
    assert gate["invalid_reasons"][str(old_ledger)] == "missing_health_feedback"
    assert summary["ledger_count"] == 0
    assert summary["invalid_reasons"][str(old_ledger)] == "missing_health_feedback"


def test_final_readiness_rejects_cross_version_ledger_without_latest_health_report(tmp_path):
    old_report_ledger = tmp_path / "reports" / "old-report-ledger.json"
    dump_json(
        {
            "schema_version": "version-ledger-v1",
            "baseline_version": "v1",
            "version_order": ["v1", "v2"],
            "summary": {
                "version_count": 2,
                "family_count": 1,
                "transition_counts": {"new": 1},
                "new_family_count": 1,
                "fixed_family_count": 0,
                "regression_family_count": 0,
                "persistent_family_count": 0,
                "health_observation_count": 2,
            },
            "health": {
                "schema_version": "version-ledger-health-v1",
                "observation_count": 2,
                "health_observation_count": 2,
            },
            "families": [
                {
                    "family": "new_root@engine",
                    "status": "new",
                    "observations": [
                        {"version_id": "v1", "state": "absent", "count": 0},
                        {"version_id": "v2", "state": "present", "count": 1},
                    ],
                }
            ],
        },
        old_report_ledger,
    )
    manifest = _write_manifest(
        tmp_path,
        name="validation-old-report-ledger",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={"version_ledger_file": str(old_report_ledger)},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["cross_version_continual_learning"]
    summary = audit["summary"]["cross_version_ledger"]
    assert gate["passed"] is False
    assert gate["invalid_reasons"][str(old_report_ledger)] == "missing_health_feedback_report"
    assert summary["ledger_count"] == 0


def test_final_readiness_consumes_postprocess_version_ledger_manifest_without_run_gates(tmp_path):
    ledger = _write_version_ledger(tmp_path)
    manifest = tmp_path / "reports" / "experiment-final-version-ledger.json"
    dump_json(
        {
            "schema_version": "version-ledger-evidence-manifest-v1",
            "evidence_mode": "comparison",
            "evidence_kind": "postprocess_ledger",
            "target_suite": "cross_version",
            "target_suites": ["cross_version"],
            "runs": [
                {
                    "target_suite": "cross_version",
                    "preset": "version_ledger",
                    "seed": "",
                    "run_file": str(tmp_path / "runs" / "run-v2.jsonl"),
                    "evidence_mode": "comparison",
                    "evidence_kind": "postprocess_ledger",
                    "version_ledger_file": str(ledger),
                }
            ],
            "experiment_meta": {
                "matrix_id": "baseline_scope_comparison",
                "comparison_group": "cross_version_continual_learning",
                "variant": {
                    "variant_id": "version_ledger",
                    "comparison_role": "support",
                    "component_focus": "cross_version_continual_learning",
                },
            },
        },
        manifest,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["cross_version_continual_learning"]["passed"] is True
    assert gates["paper_run_journal"]["passed"] is True
    assert gates["stage_level_profiling"]["passed"] is True
    assert audit["summary"]["comparison_runs"] == 0
    assert audit["summary"]["cross_version_ledger"]["ledger_count"] == 1


def test_final_readiness_audits_continual_learning_absorption_when_declared(tmp_path):
    ledger = _write_version_ledger(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        name="comparison-adaptive-continual",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="guided",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_updates={"version_ledger_file": str(ledger)},
        experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
            target_suites=("embedded_sql",),
            preset="guided",
        ),
    )
    manifest_payload = load_json(manifest)
    manifest_payload["schedule"] = "adaptive"
    manifest_payload["adaptive_config"] = {
        "continual_learning_sources": [
            {
                "path": str(ledger),
                "loaded": True,
                "family_count": 2,
                "feature_count": 4,
            }
        ]
    }
    manifest_payload["adaptive_learning"] = {
        "schema_version": "adaptive-learning-v1",
        "continual_priority_memory": {
            "imported_ledger_count": 0,
            "imported_family_count": 0,
            "family_priorities": {},
            "feature_counts": {},
        },
    }
    dump_json(manifest_payload, manifest)

    thresholds = ReadinessThresholds(
        min_live_duration_hours=0.0,
        min_live_candidate_families=0,
        min_confirmed_live_families=0,
        min_historical_confirmed=0,
        require_validation=False,
        require_seeded=False,
        require_ablation=False,
        require_comparison=False,
        require_transferability_scope=False,
        require_adaptive_component_ablation=False,
        require_runtime_efficiency=False,
        require_closed_loop_state_persistence=False,
    )
    policy = ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=())

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["continual_learning_absorption"]

    assert gate["passed"] is False
    assert gate["missing"] == ["comparison:embedded_sql:guided:loaded_sources_not_absorbed"]

    manifest_payload["adaptive_learning"]["continual_priority_memory"] = {
        "imported_ledger_count": 1,
        "imported_family_count": 2,
        "family_priorities": {"new_family@engine": 0.9, "persistent_family@engine": 0.55},
        "feature_counts": {"family:new_family": 1, "family:persistent_family": 1},
    }
    dump_json(manifest_payload, manifest)

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["continual_learning_absorption"]

    assert gate["passed"] is True
    assert gate["absorbed_run_count"] == 1
    assert gate["max_imported_ledger_count"] == 1
    assert gate["max_imported_family_count"] == 2


def test_final_readiness_requires_transferability_beyond_primary_dataframe_family(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="dataframe-only-live",
        evidence_mode="live",
        target_suite="dataframe",
        preset="live",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_discovery_responsiveness=False,
            require_target_version_audit=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("dataframe",),
            required_live_families=("dataframe",),
            required_transfer_families=("dataframe", "embedded_sql"),
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["transferability_scope"]["passed"] is False
    assert gates["transferability_scope"]["families"] == ["dataframe"]
    assert gates["transferability_scope"]["missing"] == ["embedded_sql"]


def test_final_readiness_transferability_scope_passes_with_second_target_family(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="dataframe-live",
            evidence_mode="live",
            target_suite="dataframe",
            preset="live",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        ),
        _write_manifest(
            tmp_path,
            name="embedded-sql-comparison",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=2,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_adaptive_component_ablation=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("dataframe",),
            required_live_families=("dataframe",),
            required_transfer_families=("dataframe", "embedded_sql"),
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["transferability_scope"]["passed"] is True
    assert audit["summary"]["transferability_scope"]["families"] == ["dataframe", "embedded_sql"]
    assert audit["summary"]["transferability_scope"]["non_primary_families"] == ["embedded_sql"]


def test_final_readiness_reports_missing_a_level_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(min_live_duration_hours=0.0),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert audit["ready"] is False
    assert gates["short_validation"]["passed"] is False
    assert gates["live_suite_breadth"]["passed"] is False
    assert gates["latest_confirmed_bug_families"]["passed"] is False
    assert gates["historical_confirmed_replay"]["passed"] is False
    assert gates["seeded_sensitivity"]["passed"] is False
    assert gates["module_ablation"]["passed"] is False
    assert gates["baseline_comparison"]["passed"] is False


def test_final_readiness_keeps_validation_runs_out_of_live_bug_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-with-candidate",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=3,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "validation_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["short_validation"]["passed"] is True
    assert audit["summary"]["validation_runs"] == 1
    assert audit["summary"]["validation_cases"] == 1
    assert audit["summary"]["explicit_live_runs"] == 0
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["ignored_evidence_runs"] == 0


def test_final_readiness_keeps_support_tracks_out_of_live_bug_evidence(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="ablation-baseline",
            evidence_mode="ablation",
            target_suite="core",
            preset="baseline",
            seed=3,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
                target_suites=("core",),
                preset="baseline",
            ),
        ),
        _write_manifest(
            tmp_path,
            name="ablation-with-candidate",
            evidence_mode="ablation",
            target_suite="core",
            preset="no_normalizer",
            seed=4,
            findings=[
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "ablation_candidate",
                    "suspicious_backends": ["datafusion"],
                    "discovery_origin": "organic",
                }
            ],
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
                target_suites=("core",),
                preset="no_normalizer",
            ),
        ),
        _write_manifest(
            tmp_path,
            name="comparison-baseline",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=5,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="baseline",
            ),
        ),
        _write_manifest(
            tmp_path,
            name="comparison-with-candidate",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="guided",
            seed=6,
            findings=[
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "comparison_candidate",
                    "suspicious_backends": ["duckdb"],
                    "discovery_origin": "organic",
                }
            ],
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="guided",
            ),
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["baseline_comparison"]["passed"] is True
    assert audit["summary"]["ablation_runs"] == 2
    assert audit["summary"]["comparison_runs"] == 2
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["ignored_evidence_runs"] == 0


def test_final_readiness_keeps_support_comparison_rows_out_of_baseline_gate(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="comparison-baseline-scope",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=51,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="baseline",
            ),
        ),
        _write_manifest(
            tmp_path,
            name="comparison-guided-scope",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="guided",
            seed=52,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            experiment_meta=FINAL_COMPARISON_MATRIX.command_experiment_meta(
                target_suites=("embedded_sql",),
                preset="guided",
            ),
        ),
        _write_manifest(
            tmp_path,
            name="comparison-support-cross-version",
            evidence_mode="comparison",
            target_suite="datafusion_cross",
            preset="version_ledger",
            seed=53,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": FINAL_COMPARISON_MATRIX.id,
                "comparison_group": "cross_version_continual_learning",
                "variant_id": "version_ledger",
                "comparison_role": "support",
                "component_focus": "cross_version_continual_learning",
                "analysis_tags": ["comparison", "support"],
            },
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_transferability_scope=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["baseline_comparison"]
    assert gate["passed"] is True
    assert gate["matrix_ids"] == ["baseline_scope_comparison"]
    assert gate["reference_run_count"] == 1
    assert gate["contrast_run_count"] == 1
    assert gate["missing_reference_groups"] == []
    assert gate["missing_contrast_groups"] == []
    assert audit["summary"]["comparison_runs"] == 2
    assert audit["summary"]["comparison_suites"] == ["embedded_sql"]
    assert audit["summary"]["baseline_comparison"]["group_count"] == 1


def test_final_readiness_run_rows_include_canonical_comparison_role(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="comparison-canonical-role",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="scope_variant",
        seed=9,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "matrix_id": "baseline_scope_comparison",
        "comparison_group": "scope_comparison",
        "analysis_tags": ["comparison"],
        "variant_by_preset": {
            "scope_variant": {
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["scope"],
            }
        },
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
        ),
    )

    run = next(item for item in audit["runs"] if item["evidence_mode"] == "comparison")
    assert run["comparison_role"] == "contrast"
    assert run["canonical_comparison_role"] == "contrast"
    assert run["variant_label"] == "scope_variant"
    assert run["variant_group_id"] == "embedded_sql|scope_comparison|baseline_scope_comparison"


def test_final_readiness_requires_structured_experiment_identity(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-structured-identity",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="guided",
        seed=10,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "matrix_id": "",
        "comparison_group": "",
        "analysis_tags": ["comparison"],
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_experiment_identity"]["passed"] is False
    assert gates["structured_experiment_identity"]["missing"] == ["comparison:embedded_sql:guided"]


def test_final_readiness_requires_stage_level_profiling(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-stage-profile",
        evidence_mode="seeded",
        target_suite="seeded_filter",
        preset="guided_filter",
        seed=11,
        include_stage_profile=False,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["stage_level_profiling"]["passed"] is False
    assert gates["stage_level_profiling"]["missing"] == ["seeded:seeded_filter:guided_filter"]


def test_final_readiness_rejects_low_throughput_runtime_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="low-throughput-runtime",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=17,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_path = run_meta_path(run_file)
    meta = final_readiness.load_json(meta_path)
    meta["throughput_cases_s"] = 0.0
    final_readiness.dump_json(meta, meta_path)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_transferability_scope=False,
            min_throughput_cases_s=0.1,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["runtime_efficiency"]["passed"] is False
    assert gates["runtime_efficiency"]["issues"] == [
        "validation:datafusion_cross:validation_smoke:throughput_below_min"
    ]
    assert audit["summary"]["runtime_efficiency"]["min_throughput_cases_s"] == 0.0


def test_final_readiness_rejects_high_scheduler_feedback_share(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="high-scheduler-feedback",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=18,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_path = run_meta_path(run_file)
    meta = final_readiness.load_json(meta_path)
    meta["stage_profile"]["totals_ms"]["scheduler_feedback_ms"] = 9.0
    meta["stage_profile"]["totals_ms"]["total_case_wall_ms"] = 10.0
    final_readiness.dump_json(meta, meta_path)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_transferability_scope=False,
            max_scheduler_feedback_share=0.5,
            min_scheduler_feedback_cases=1,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["runtime_efficiency"]["passed"] is False
    assert gates["runtime_efficiency"]["issues"] == [
        "validation:datafusion_cross:validation_smoke:scheduler_feedback_share_high"
    ]
    assert audit["summary"]["runtime_efficiency"]["max_scheduler_feedback_share"] == 0.9


def test_final_readiness_skips_scheduler_feedback_share_for_short_smoke_runs(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="short-scheduler-feedback-smoke",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=181,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_path = run_meta_path(run_file)
    meta = final_readiness.load_json(meta_path)
    meta["executed_cases"] = 1
    meta["stage_profile"]["totals_ms"]["scheduler_feedback_ms"] = 9.0
    meta["stage_profile"]["totals_ms"]["total_case_wall_ms"] = 10.0
    final_readiness.dump_json(meta, meta_path)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_transferability_scope=False,
            max_scheduler_feedback_share=0.5,
            min_scheduler_feedback_cases=10,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["runtime_efficiency"]["passed"] is True
    assert gates["runtime_efficiency"]["issues"] == []
    summary = audit["summary"]["runtime_efficiency"]
    assert summary["max_scheduler_feedback_share"] == 0.0
    assert summary["scheduler_feedback_share_run_count"] == 0
    assert summary["scheduler_feedback_share_skipped_run_count"] == 1
    assert summary["scheduler_feedback_share_skipped_runs"] == [
        "validation:datafusion_cross:validation_smoke:cases_below_scheduler_share_min:1/10"
    ]


def test_final_readiness_exempts_reducer_ablation_from_scheduler_feedback_share_gate(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="reducer-ablation-scheduler-feedback",
        evidence_mode="ablation",
        target_suite="core",
        preset="reducer",
        seed=182,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        experiment_meta=FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
            target_suites=("core",),
            preset="reducer",
        ),
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_path = run_meta_path(run_file)
    meta = final_readiness.load_json(meta_path)
    meta["executed_cases"] = 100
    meta["stage_profile"]["totals_ms"]["scheduler_feedback_ms"] = 90.0
    meta["stage_profile"]["totals_ms"]["total_case_wall_ms"] = 100.0
    final_readiness.dump_json(meta, meta_path)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_transferability_scope=False,
            require_cross_version_ledger=False,
            max_scheduler_feedback_share=0.5,
            min_scheduler_feedback_cases=1,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["runtime_efficiency"]["passed"] is True
    assert gates["runtime_efficiency"]["issues"] == []
    summary = audit["summary"]["runtime_efficiency"]
    assert summary["max_scheduler_feedback_share"] == 0.0
    assert summary["scheduler_feedback_share_run_count"] == 0
    assert summary["scheduler_feedback_share_skipped_runs"] == [
        "ablation:core:reducer:scheduler_feedback_share_exempt:reducer_ablation"
    ]


def test_final_readiness_requires_live_discovery_responsiveness_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-without-first-candidate",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("query_engine",),
            required_final_matrix_ids=(),
        ),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["discovery_responsiveness"]
    assert gate["passed"] is False
    assert gate["observed_run_count"] == 0


def test_final_readiness_derives_discovery_responsiveness_from_run_log(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-with-log-candidate",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "log_root",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
            max_first_candidate_elapsed_s=1.0,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("query_engine",),
            required_final_matrix_ids=(),
        ),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["discovery_responsiveness"]
    assert gate["passed"] is True
    assert gate["best_first_candidate_case_index"] == 0
    assert gate["best_first_candidate_elapsed_s"] == 0.1
    assert gate["avg_candidate_bug_discovery_auc"] == 1.0


def test_final_readiness_rejects_slow_first_candidate_discovery(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="slow-first-candidate",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=42,
        first_candidate_elapsed_s=7200.0,
        candidate_bug_discovery_auc=0.1,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
            max_first_candidate_elapsed_s=60.0,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("query_engine",),
            required_final_matrix_ids=(),
        ),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["discovery_responsiveness"]
    assert gate["passed"] is False
    assert gate["best_first_candidate_elapsed_s"] == 7200.0


def test_final_readiness_requires_closed_loop_state_persistence_for_live_runs(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-without-closed-loop-state",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
        include_closed_loop_state=False,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("query_engine",),
            required_final_matrix_ids=(),
        ),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["closed_loop_state_persistence"]
    assert gate["passed"] is False
    assert gate["missing"] == ["live:datafusion_cross:live_datafusion:missing_state_file"]


def test_final_readiness_requires_adaptive_learning_health_for_live_closed_loop_state(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-with-legacy-closed-loop-state",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
    )
    run_file = tmp_path / "runs" / "run-live-with-legacy-closed-loop-state.jsonl"
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path)
    meta["closed_loop_state_summary"] = {"seen_signature_count": 1}
    dump_json(meta, meta_path)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("query_engine",),
            required_final_matrix_ids=(),
        ),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["closed_loop_state_persistence"]
    assert gate["passed"] is False
    assert gate["missing"] == []
    assert gate["weak"] == []
    assert gate["missing_health"] == [
        "live:datafusion_cross:live_datafusion:missing_adaptive_learning_health"
    ]


def test_final_readiness_requires_component_level_adaptive_live_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-adaptive-without-component-evidence",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
        run_updates={
            "adaptive_components": _adaptive_component_flags(local_source_scheduler=True),
            "disabled_adaptive_components": [],
        },
    )
    run_file = tmp_path / "runs" / "run-live-adaptive-without-component-evidence.jsonl"
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path)
    meta["closed_loop_state_summary"]["adaptive_learning_health"] = {
        "schema_version": "adaptive-learning-health-v1",
        "bandit_count": 1,
        "arm_count": 1,
        "total_pulls": 1,
        "avg_health_penalty": 0.0,
        "max_health_penalty": 0.0,
        "avg_uncertainty": 1.0,
        "exploration_memory": {
            "total_records": 0,
            "context_count": 0,
            "action_count": 0,
        },
    }
    meta["closed_loop_state_summary"].pop("quality_archive_health", None)
    meta["closed_loop_state_summary"].pop("seed_quota_health", None)
    meta["closed_loop_state_summary"].pop("champion_corpus_health", None)
    meta.pop("lhs_seeding", None)
    meta.pop("champion_corpus", None)
    dump_json(meta, meta_path)
    manifest_payload = final_readiness.load_json(manifest)
    manifest_payload["adaptive_state"] = [
        {
            "arm_id": "datafusion_cross:live_datafusion:0",
            "target_suite": "datafusion_cross",
            "preset": "live_datafusion",
            "pulls": 1,
            "learning_signal": 0.1,
            "annealing_temperature": 0.0,
            "bayesian_exploration_observation_count": 0,
        }
    ]
    final_readiness.dump_json(manifest_payload, manifest)

    thresholds = ReadinessThresholds(
        min_live_duration_hours=0.0,
        min_live_candidate_families=0,
        min_confirmed_live_families=0,
        min_historical_confirmed=0,
        require_validation=False,
        require_seeded=False,
        require_ablation=False,
        require_comparison=False,
        require_transferability_scope=False,
        require_adaptive_component_ablation=False,
        require_cross_version_ledger=False,
        require_runtime_efficiency=False,
    )
    policy = ReadinessPolicy(
        required_live_suites=("datafusion_cross",),
        required_live_families=("query_engine",),
        required_final_matrix_ids=(),
    )

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]

    assert gate["passed"] is False
    assert gate["proven_components"] == ["backend_pair_learning", "scheduler_learning"]
    assert gate["missing"] == [
        "live:datafusion_cross:live_datafusion:active_learning",
        "live:datafusion_cross:live_datafusion:bayesian_exploration",
        "live:datafusion_cross:live_datafusion:bd_axis_bandit",
        "live:datafusion_cross:live_datafusion:champion_corpus",
        "live:datafusion_cross:live_datafusion:hierarchical_archive",
        "live:datafusion_cross:live_datafusion:lhs_seeding",
        "live:datafusion_cross:live_datafusion:online_reward_model",
        "live:datafusion_cross:live_datafusion:quality_archive",
        "live:datafusion_cross:live_datafusion:runtime_cost_learning",
        "live:datafusion_cross:live_datafusion:scheduler_annealing",
        "live:datafusion_cross:live_datafusion:seed_energy_tier",
        "live:datafusion_cross:live_datafusion:seed_quota",
        "live:datafusion_cross:live_datafusion:value_catalog",
    ]
    assert gate["optional_missing"] == [
        "live:datafusion_cross:live_datafusion:champion_graft_donor",
        "live:datafusion_cross:live_datafusion:continual_learning",
    ]
    assert gate["unproven_optional_components"] == [
        "champion_graft_donor",
        "continual_learning",
    ]
    assert "hierarchical_archive" in gate["unproven_required_components"]

    meta["closed_loop_state_summary"]["adaptive_learning_health"].update(
        {
            "reward_model_update_count": 1,
            "reward_model_feature_count": 2,
            "version_memory_key_count": 1,
            "runtime_cost_observation_count": 1,
            "runtime_cost_total": 0.1,
            "value_catalog_entry_pulls": 1,
            "value_catalog_entry_arm_count": 1,
            "bd_axis_weight_pulls": 1,
            "bd_axis_weight_arm_count": 1,
            "seed_energy_tier_pulls": 1,
            "seed_energy_tier_arm_count": 1,
            "champion_graft_donor_pulls": 1,
            "champion_graft_donor_arm_count": 1,
            "continual_priority_memory": {
                "imported_ledger_count": 1,
                "imported_family_count": 1,
                "family_count": 1,
                "feature_count": 2,
            },
            "exploration_memory": {
                "total_records": 1,
                "context_count": 2,
                "action_count": 1,
            },
        }
    )
    meta["closed_loop_state_summary"]["quality_archive_health"] = {
        "schema_version": "quality-archive-health-v1",
        "cell_count": 1,
        "hierarchical_enabled": True,
        "child_cell_count": 1,
        "split_cell_count": 1,
        "seed_count": 1,
        "elite_seed_count": 1,
        "reward_count": 1,
        "outcome_count": 1,
        "invalid_count": 0,
        "fallback_count": 0,
        "false_positive_count": 0,
    }
    meta["closed_loop_state_summary"]["seed_quota_health"] = {
        "enabled": True,
        "active": True,
        "cluster_count": 1,
        "seed_count": 1,
    }
    meta["closed_loop_state_summary"]["champion_corpus_health"] = {
        "enabled": True,
        "family_hit_count": 1,
        "promoted_family_count": 1,
        "version_id": "test-version",
    }
    meta["lhs_seeding"] = {"enabled": True, "sample_count": 256}
    meta["champion_corpus"] = {
        "enabled": True,
        "path": str(tmp_path / "runs" / "champion_corpus.jsonl"),
        "version_id": "test-version",
        "injected_count": 1,
    }
    dump_json(meta, meta_path)
    manifest_payload["adaptive_state"][0]["annealing_temperature"] = 0.2
    manifest_payload["adaptive_state"][0]["bayesian_exploration_observation_count"] = 1
    final_readiness.dump_json(manifest_payload, manifest)

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]

    assert gate["passed"] is True
    assert gate["missing"] == []
    assert gate["optional_missing"] == []
    assert gate["unproven_declared_components"] == []
    assert gate["adaptive_selection_total_count"] == 5
    assert gate["adaptive_selection_scopes"] == [
        "backend_pair",
        "generator_profile",
        "metamorphic_relation",
        "semantic_objective",
        "version_pair",
    ]
    assert gate["proven_components"] == [
        "active_learning",
        "backend_pair_learning",
        "bayesian_exploration",
        "bd_axis_bandit",
        "champion_corpus",
        "champion_graft_donor",
        "continual_learning",
        "hierarchical_archive",
        "lhs_seeding",
        "online_reward_model",
        "quality_archive",
        "runtime_cost_learning",
        "scheduler_annealing",
        "scheduler_learning",
        "seed_energy_tier",
        "seed_quota",
        "value_catalog",
    ]


def test_final_readiness_requires_adaptive_selection_telemetry_for_live_adaptive_runs(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-adaptive-without-selection-telemetry",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
        include_adaptive_selection=False,
        run_updates={
            "adaptive_components": _adaptive_component_flags(local_source_scheduler=True),
            "disabled_adaptive_components": [],
        },
    )
    thresholds = ReadinessThresholds(
        min_live_duration_hours=0.0,
        min_live_candidate_families=0,
        min_confirmed_live_families=0,
        min_historical_confirmed=0,
        require_validation=False,
        require_seeded=False,
        require_ablation=False,
        require_comparison=False,
        require_transferability_scope=False,
        require_adaptive_component_ablation=False,
        require_cross_version_ledger=False,
        require_runtime_efficiency=False,
    )
    policy = ReadinessPolicy(
        required_live_suites=("datafusion_cross",),
        required_live_families=("query_engine",),
        required_final_matrix_ids=(),
    )

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]

    assert gate["passed"] is False
    assert "live:datafusion_cross:live_datafusion:adaptive_selection_telemetry" in gate["missing"]
    assert gate["adaptive_selection_total_count"] == 0


def test_final_readiness_accepts_live_adaptive_component_proven_by_another_run(tmp_path):
    adaptive_components = _adaptive_component_flags(
        local_source_scheduler=True,
        champion_graft_donor=False,
        continual_learning=False,
    )
    missing_hierarchical = _write_manifest(
        tmp_path,
        name="live-adaptive-missing-hierarchical",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
        run_updates={
            "adaptive_components": adaptive_components,
            "disabled_adaptive_components": [],
        },
    )
    proven_hierarchical = _write_manifest(
        tmp_path,
        name="live-adaptive-proven-hierarchical",
        evidence_mode="live",
        target_suite="polars_cross",
        preset="live_datafusion",
        seed=2,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
        run_updates={
            "adaptive_components": adaptive_components,
            "disabled_adaptive_components": [],
        },
    )
    run_file = Path(final_readiness.load_json(missing_hierarchical)["runs"][0]["run_file"])
    meta_path = run_meta_path(run_file)
    meta = final_readiness.load_json(meta_path)
    quality = meta["closed_loop_state_summary"]["quality_archive_health"]
    quality["hierarchical_enabled"] = True
    quality["child_cell_count"] = 0
    quality["split_cell_count"] = 0
    final_readiness.dump_json(meta, meta_path)

    audit = build_final_readiness(
        [missing_hierarchical, proven_hierarchical],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_transferability_scope=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_runtime_efficiency=False,
        ),
        policy=ReadinessPolicy(required_live_suites=(), required_live_families=(), required_final_matrix_ids=()),
    )

    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]
    summary = audit["summary"]["adaptive_live_component_evidence"]

    assert gate["passed"] is True
    assert "hierarchical_archive" in gate["proven_components"]
    assert not any(item.endswith(":hierarchical_archive") for item in gate["missing"])
    assert gate["unproven_required_components"] == []
    assert gate["unproven_optional_components"] == []
    assert any(
        "hierarchical_archive" in row["missing_components"]
        for row in summary["rows"]
        if row["label"] == "live:datafusion_cross:live_datafusion"
    )


def test_final_readiness_infers_adaptive_selection_requirement_from_run_config(tmp_path):
    config = {
        "enable_replay_bug": False,
        "generator_profile_learning_weight": 1.0,
        "semantic_objective_learning_weight": 1.0,
        "metamorphic_relation_learning_weight": 1.0,
        "version_pair_learning_weight": 1.0,
        "backend_pair_learning_weight": 1.0,
        "enable_generator_profile_learning": True,
        "enable_semantic_objective_learning": True,
        "enable_metamorphic_relation_learning": True,
        "enable_backend_pair_learning": True,
        "enable_metamorphic_oracle": True,
        "enable_quality_archive": False,
    }
    manifest = _write_manifest(
        tmp_path,
        name="live-adaptive-config-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config=config,
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        first_candidate_case_index=0,
        first_candidate_elapsed_s=0.1,
    )
    thresholds = ReadinessThresholds(
        min_live_duration_hours=0.0,
        min_live_candidate_families=0,
        min_confirmed_live_families=0,
        min_historical_confirmed=0,
        require_validation=False,
        require_seeded=False,
        require_ablation=False,
        require_comparison=False,
        require_transferability_scope=False,
        require_adaptive_component_ablation=False,
        require_cross_version_ledger=False,
        require_runtime_efficiency=False,
    )
    policy = ReadinessPolicy(
        required_live_suites=("datafusion_cross",),
        required_live_families=("query_engine",),
        required_final_matrix_ids=(),
    )

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]

    assert gate["passed"] is True
    assert gate["declared_components"] == ["backend_pair_learning", "scheduler_learning"]
    assert gate["adaptive_selection_total_count"] == 5
    assert gate["adaptive_selection_scopes"] == [
        "backend_pair",
        "generator_profile",
        "metamorphic_relation",
        "semantic_objective",
        "version_pair",
    ]

    run_file = tmp_path / "runs" / "run-live-adaptive-config-only.jsonl"
    run_file.write_text(
        '{"case_index": 0, "elapsed_s": 0.1, "case": {"case_id": "case-live-adaptive-config-only", "seed": 1, "program": {"operations": []}}, "findings": []}\n',
        encoding="utf-8",
    )

    audit = build_final_readiness([manifest], thresholds=thresholds, policy=policy)
    gate = {gate["name"]: gate for gate in audit["gates"]}["adaptive_live_component_evidence"]

    assert gate["passed"] is False
    assert gate["missing"] == [
        "live:datafusion_cross:live_datafusion:adaptive_selection_telemetry",
        "live:datafusion_cross:live_datafusion:backend_pair_learning",
    ]


def test_final_readiness_requires_paper_run_journal_coverage(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-paper-journal",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=41,
        write_paper_run_journal=False,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["paper_run_journal"]["passed"] is False
    assert gates["paper_run_journal"]["missing"] == ["validation:datafusion_cross:validation_smoke"]
    assert audit["summary"]["paper_run_journal_covered_runs"] == 0
    assert audit["summary"]["paper_run_journal_missing_runs"] == ["validation:datafusion_cross:validation_smoke"]


def test_final_readiness_prefers_structured_matrix_identity_for_support_tracks(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="ablation-baseline-structured",
            evidence_mode="ablation",
            target_suite="core",
            preset="stable_base",
            seed=10,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "variant_id": "baseline",
                "comparison_role": "baseline",
                "analysis_tags": ["ablation", "baseline"],
            },
        ),
        _write_manifest(
            tmp_path,
            name="ablation-structured",
            evidence_mode="ablation",
            target_suite="core",
            preset="focus_variant",
            seed=11,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "variant_id": "focus_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "component_focus": "semantic_normalizer",
                "analysis_tags": ["ablation", "noise_control"],
            },
        ),
        _write_manifest(
            tmp_path,
            name="comparison-baseline-structured",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="stable_base",
            seed=13,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "baseline_scope_comparison",
                "comparison_group": "scope_comparison",
                "variant_id": "baseline",
                "comparison_role": "baseline",
                "analysis_tags": ["comparison", "baseline"],
            },
        ),
        _write_manifest(
            tmp_path,
            name="comparison-structured",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="scope_variant",
            seed=12,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "baseline_scope_comparison",
                "comparison_group": "scope_comparison",
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["comparison", "scope"],
            },
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["module_ablation"]["matrix_ids"] == ["module_ablation"]
    assert gates["module_ablation"]["variants"] == ["baseline", "focus_variant"]
    assert gates["module_ablation"]["reference_run_count"] == 1
    assert gates["module_ablation"]["contrast_run_count"] == 1
    assert gates["baseline_comparison"]["passed"] is True
    assert gates["baseline_comparison"]["matrix_ids"] == ["baseline_scope_comparison"]
    assert gates["baseline_comparison"]["variants"] == ["baseline", "scope_variant"]
    assert gates["baseline_comparison"]["reference_run_count"] == 1
    assert gates["baseline_comparison"]["contrast_run_count"] == 1
    assert audit["summary"]["ablation_matrix_ids"] == ["module_ablation"]
    assert audit["summary"]["ablation_variants"] == ["baseline", "focus_variant"]
    assert audit["summary"]["comparison_matrix_ids"] == ["baseline_scope_comparison"]
    assert audit["summary"]["comparison_variants"] == ["baseline", "scope_variant"]


def test_final_readiness_backfills_support_track_identity_from_experiment_meta(tmp_path):
    ablation_baseline_manifest = _write_manifest(
        tmp_path,
        name="ablation-meta-baseline",
        evidence_mode="ablation",
        target_suite="core",
        preset="stable_base",
        seed=20,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    ablation_manifest = _write_manifest(
        tmp_path,
        name="ablation-meta-only",
        evidence_mode="ablation",
        target_suite="core",
        preset="focus_variant",
        seed=21,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    comparison_baseline_manifest = _write_manifest(
        tmp_path,
        name="comparison-meta-baseline",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="stable_base",
        seed=23,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    comparison_manifest = _write_manifest(
        tmp_path,
        name="comparison-meta-only",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="scope_variant",
        seed=22,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    ablation_payload = final_readiness.load_json(ablation_manifest)
    ablation_payload["experiment_meta"] = {
        "matrix_id": "module_ablation",
        "comparison_group": "module_ablation",
        "analysis_tags": ["ablation"],
        "variant_by_preset": {
            "stable_base": {
                "variant_id": "baseline",
                "comparison_role": "baseline",
                "analysis_tags": ["baseline"],
            },
            "focus_variant": {
                "variant_id": "focus_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "component_focus": "semantic_normalizer",
                "analysis_tags": ["noise_control"],
            }
        },
    }
    final_readiness.dump_json(ablation_payload, ablation_manifest)
    ablation_baseline_payload = final_readiness.load_json(ablation_baseline_manifest)
    ablation_baseline_payload["experiment_meta"] = ablation_payload["experiment_meta"]
    final_readiness.dump_json(ablation_baseline_payload, ablation_baseline_manifest)

    comparison_payload = final_readiness.load_json(comparison_manifest)
    comparison_payload["experiment_meta"] = {
        "matrix_id": "baseline_scope_comparison",
        "comparison_group": "scope_comparison",
        "analysis_tags": ["comparison"],
        "variant_by_preset": {
            "stable_base": {
                "variant_id": "baseline",
                "comparison_role": "baseline",
                "analysis_tags": ["baseline"],
            },
            "scope_variant": {
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["scope"],
            }
        },
    }
    final_readiness.dump_json(comparison_payload, comparison_manifest)
    comparison_baseline_payload = final_readiness.load_json(comparison_baseline_manifest)
    comparison_baseline_payload["experiment_meta"] = comparison_payload["experiment_meta"]
    final_readiness.dump_json(comparison_baseline_payload, comparison_baseline_manifest)

    audit = build_final_readiness(
        [ablation_baseline_manifest, ablation_manifest, comparison_baseline_manifest, comparison_manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["baseline_comparison"]["passed"] is True
    assert audit["summary"]["ablation_matrix_ids"] == ["module_ablation"]
    assert audit["summary"]["ablation_variants"] == ["baseline", "focus_variant"]
    assert audit["summary"]["comparison_matrix_ids"] == ["baseline_scope_comparison"]
    assert audit["summary"]["comparison_variants"] == ["baseline", "scope_variant"]


def test_final_readiness_records_registered_structured_semantic_focus(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-structured-semantic-focus",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=31,
        config={
            "enable_replay_bug": False,
            "guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
            ],
            "semantic_focus_families": [
                "materialization_boundary",
                "string_semantics",
                "join_membership",
                "set_semantics",
            ],
            "semantic_focus_signals": [
                "common_api_workflow",
                "filter_input_materialization",
                "distinct_input_materialization",
                "case_when_membership",
                "string_pattern_case_when",
            ],
            "effective_guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
                "semantic_family:materialization_boundary",
                "semantic_family:string_semantics",
                "semantic_family:join_membership",
                "semantic_family:set_semantics",
                "semantic_signal:common_api_workflow",
                "semantic_signal:filter_input_materialization",
                "semantic_signal:distinct_input_materialization",
                "semantic_signal:case_when_membership",
                "semantic_signal:string_pattern_case_when",
            ],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_semantic_focus"]["passed"] is True
    assert gates["effective_guidance_targets"]["passed"] is True
    run = next(item for item in audit["runs"] if item["preset"] == "live_common_api_workflow_metamorphic")
    assert "materialization_boundary" in run["semantic_focus_families"]
    assert "common_api_workflow" in run["semantic_focus_signals"]
    assert "materialization_boundary" in audit["summary"]["semantic_focus_families"]
    assert "common_api_workflow" in audit["summary"]["semantic_focus_signals"]


def test_final_readiness_summarizes_semantic_contract_and_ir_rewrite_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-semantic-contract-ir",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=41,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(load_json(manifest)["runs"][0]["run_file"])
    run_file.write_text("", encoding="utf-8")
    append_jsonl(
        {
            "case_index": 0,
            "elapsed_s": 0.1,
            "case": {"case_id": "case-contract-ir", "seed": 41, "program": {"operations": []}},
            "semantic_contract_lattice": {
                "schema_version": "semantic-contract-lattice-v1",
                "case_id": "case-contract-ir",
                "boundary_axes": ["ordering"],
                "strict_axes": ["null", "nan"],
                "operation_contracts": [{"operation": "limit"}],
            },
            "mutation": {
                "operator": "ir_pushdown_filter",
                "detail": "ir_pushdown_filter:test",
            },
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "ordering_or_limit",
                    "mismatch_class": "row_order",
                    "triage_verdict": "candidate_implementation_bug",
                    "signature": "sig-contract-ir",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        run_file,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    contract = audit["summary"]["semantic_contract_lattice"]
    rewrite = audit["summary"]["ir_rewrite_rule_evidence"]
    assert contract["contract_row_count"] == 1
    assert contract["case_row_count"] == 1
    assert contract["boundary_axes"] == ["ordering"]
    assert contract["matched_boundary_axes"] == ["ordering"]
    assert rewrite["rewrite_row_count"] == 1
    assert rewrite["rule_ids"] == ["ir.rewrite.filter_pushdown"]
    assert rewrite["semantics_classes"] == ["semantics_preserving"]
    md = final_readiness._render_markdown(audit)
    assert "Semantic contract lattice evidence" in md
    assert "IR rewrite rule evidence" in md


def test_final_readiness_flags_registered_semantic_focus_metadata_drift(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-semantic-focus-drift",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=32,
        config={
            "enable_replay_bug": False,
            "guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
            ],
            "semantic_focus_families": [
                "materialization_boundary",
                "string_semantics",
                "join_membership",
                "set_semantics",
            ],
            "semantic_focus_signals": [
                "common_api_workflow",
                "filter_input_materialization",
                "distinct_input_materialization",
                "case_when_membership",
                "string_pattern_case_when",
            ],
            "effective_guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
                "semantic_family:materialization_boundary",
                "semantic_family:string_semantics",
                "semantic_family:join_membership",
                "semantic_family:set_semantics",
                "semantic_signal:common_api_workflow",
                "semantic_signal:filter_input_materialization",
                "semantic_signal:distinct_input_materialization",
                "semantic_signal:case_when_membership",
                "semantic_signal:string_pattern_case_when",
            ],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "variant_by_preset": {
            "live_common_api_workflow_metamorphic": {
                "semantic_focus_families": ["wrong_family"],
                "semantic_focus_signals": ["wrong_signal"],
            }
        }
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_semantic_focus"]["passed"] is False
    assert gates["structured_semantic_focus"]["issues"] == [
        "validation:datafusion_cross:live_common_api_workflow_metamorphic"
    ]


def test_final_readiness_flags_effective_guidance_target_drift(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-effective-guidance-drift",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=33,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_file = run_meta_path(run_file)
    meta_payload = final_readiness.load_json(meta_file)
    meta_payload["config"]["effective_guidance_targets"] = ["wrong_target"]
    final_readiness.dump_json(meta_payload, meta_file)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["effective_guidance_targets"]["passed"] is False
    assert gates["effective_guidance_targets"]["issues"] == [
        "validation:datafusion_cross:live_common_api_workflow_metamorphic"
    ]


def test_final_readiness_summary_only_mode_does_not_claim_paper_readiness(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-summary-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=6,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "summary_only_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        scan_run_logs=False,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert audit["ready"] is False
    assert gates["run_log_scan"]["passed"] is False
    assert audit["summary"]["run_logs_scanned"] == 0
    assert audit["summary"]["run_logs_scan_skipped"] == 1
    assert audit["summary"]["live_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}


def test_final_readiness_counts_legacy_historical_replay_without_run_replay_flag(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="legacy-historical-duckdb-22075",
        evidence_mode="historical",
        target_suite="cross_family",
        preset="join_groupby_stress",
        seed=22075,
        known_bug_id="duckdb-22075",
        config={},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=1,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    assert audit["summary"]["historical_confirmed_bug_ids"] == ["duckdb-22075"]
    assert audit["runs"][0]["enable_replay_bug"] is True


def test_final_readiness_does_not_default_legacy_manifest_to_live(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="legacy-with-candidate",
        evidence_mode=None,
        target_suite="datafusion_cross",
        preset="old_live_datafusion",
        seed=7,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "legacy_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["ignored_evidence_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is False


def test_final_readiness_counts_only_fresh_policy_live_runs_as_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-without-replay-filter",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="old_live_datafusion",
        seed=9,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "stale_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["explicit_live_runs"] == 1
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["replay_policy_rejected_live_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is False


def test_final_readiness_rejects_seeded_suite_as_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="seeded-mislabeled-live",
        evidence_mode="live",
        target_suite="seeded_filter",
        preset="guided_filter",
        seed=8,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "filter_predicate",
                "suspicious_backends": ["buggy_filter"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("seeded_filter",), required_live_families=("seeded_fault",)),
    )

    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["seeded_runs"] == 0
    assert audit["summary"]["ignored_evidence_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}


def test_final_readiness_excludes_known_saturated_live_families_from_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-known-family",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "grouped_topk_null_sort_key",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
                "paper_status": "confirmed_bug",
            }
        ],
        config={
            "enable_replay_bug": False,
            "known_saturated_bug_families": ["grouped_topk_null_sort_key@datafusion"],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["known_saturated_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["confirmed_live_candidate_families"] == {}


def test_final_readiness_counts_external_upstream_confirmation_without_rewarding_known_family(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-known-family",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "grouped_topk_null_sort_key",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
                "paper_status": "candidate_bug_needs_external_confirmation",
            }
        ],
        config={
            "enable_replay_bug": False,
            "known_saturated_bug_families": ["grouped_topk_null_sort_key@datafusion"],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    confirmation_file = tmp_path / "latest_confirmations.json"
    dump_json(
        {
            "schema_version": 1,
            "confirmations": [
                {
                    "family": "grouped_topk_null_sort_key@datafusion",
                    "issue_url": "https://github.com/apache/datafusion/issues/22190",
                    "discovery_credit": "datadiff_submitted",
                    "upstream_status": "upstream_labeled_bug",
                    "labels": ["bug"],
                },
                {
                    "family": "similar_existing@pandas",
                    "issue_url": "https://github.com/pandas-dev/pandas/issues/63527",
                    "discovery_credit": "similar_existing",
                    "upstream_status": "upstream_labeled_bug",
                    "labels": ["Bug"],
                }
            ],
        },
        confirmation_file,
    )

    audit = build_final_readiness(
        [manifest],
        latest_confirmation_files=[confirmation_file],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=1,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["external_confirmed_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert audit["summary"]["confirmed_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["latest_confirmed_bug_families"] is True


def test_final_readiness_policy_keeps_top_level_requirements_out_of_engine(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
            require_adaptive_component_ablation=False,
            require_cross_version_ledger=False,
            require_discovery_responsiveness=False,
            require_target_version_audit=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("dataframe", "embedded_sql", "query_engine"),
            required_final_matrix_ids=(),
        ),
    )

    assert audit["ready"] is True
    assert audit["policy"]["required_live_suites"] == ("datafusion_cross",)


def test_analyze_final_readiness_writes_markdown_and_json(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(final_readiness, "REPORTS_DIR", reports_dir)
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    md_path, json_path = final_readiness.analyze_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    assert md_path.exists()
    assert json_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Final Experiment Readiness" in md
    assert "Adaptive learning health" in md


def test_final_readiness_default_manifest_resolution_uses_latest_limit(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    manifests = []
    for idx in range(DEFAULT_FINAL_READINESS_MANIFEST_LIMIT + 2):
        path = runs_dir / f"experiment-{idx:02d}.json"
        path.write_text('{"runs": []}\n', encoding="utf-8")
        timestamp_ns = 1_800_000_000_000_000_000 + idx
        path.touch()
        os.utime(path, ns=(timestamp_ns, timestamp_ns))
        manifests.append(path)
    monkeypatch.setattr(final_readiness, "RUNS_DIR", runs_dir)

    selected = final_readiness._resolve_manifest_files(None)

    assert selected == manifests[-DEFAULT_FINAL_READINESS_MANIFEST_LIMIT:]
    assert final_readiness._resolve_manifest_files(None, manifest_limit=None) == manifests
    assert final_readiness._resolve_manifest_files([manifests[0]], manifest_limit=1) == [manifests[0]]
    extra_manifest = tmp_path / "reports" / "experiment-final-version-ledger.json"
    extra_manifest.parent.mkdir()
    extra_manifest.write_text('{"runs": []}\n', encoding="utf-8")
    assert final_readiness._resolve_manifest_files(
        None,
        extra_manifest_files=[extra_manifest],
    ) == [*manifests[-DEFAULT_FINAL_READINESS_MANIFEST_LIMIT:], extra_manifest]
    assert final_readiness._resolve_manifest_files(
        [manifests[0]],
        extra_manifest_files=[extra_manifest],
    ) == [manifests[0], extra_manifest]


def test_final_readiness_rejects_live_runs_without_latest_frozen_authority_provenance(tmp_path, monkeypatch):
    manifest = _write_manifest(
        tmp_path,
        name="live-provenance-mismatch",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_provenance={
            "vcs": {"git_commit": "old-head", "workspace_dirty": True},
            "harness": {"authority": False, "freeze_intent": False, "latest_code_claim": False},
        },
    )
    monkeypatch.setattr(final_readiness, "current_workspace_git_commit", lambda: "new-head")

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["latest_live_provenance"]["passed"] is False
    assert sorted(gates["latest_live_provenance"]["issues"]) == [
        "live:datafusion_cross:live_datafusion:dirty_or_unknown_workspace",
        "live:datafusion_cross:live_datafusion:freeze_not_declared",
        "live:datafusion_cross:live_datafusion:head_mismatch",
        "live:datafusion_cross:live_datafusion:latest_code_not_declared",
        "live:datafusion_cross:live_datafusion:missing_git_status_artifact",
        "live:datafusion_cross:live_datafusion:missing_launcher_env_artifact",
        "live:datafusion_cross:live_datafusion:missing_manifest_artifact",
        "live:datafusion_cross:live_datafusion:missing_pip_freeze_artifact",
        "live:datafusion_cross:live_datafusion:missing_strategy_snapshot_artifact",
        "live:datafusion_cross:live_datafusion:non_authority",
    ]


def _write_manifest(
    root: Path,
    *,
    name: str,
    evidence_mode: str | None,
    target_suite: str,
    preset: str,
    seed: int,
    findings: list[dict] | None = None,
    known_bug_id: str = "",
    config: dict | None = None,
    replay_filter: dict | None = None,
    run_updates: dict | None = None,
    include_stage_profile: bool = True,
    first_candidate_case_index: int | None = None,
    first_candidate_elapsed_s: float | None = None,
    candidate_bug_discovery_auc: float | None = None,
    include_closed_loop_state: bool | None = None,
    include_adaptive_selection: bool = True,
    run_provenance: dict | None = None,
    write_paper_run_journal: bool = True,
    experiment_meta: dict | None = None,
) -> Path:
    runs_dir = root / "runs"
    run_file = runs_dir / f"run-{name}.jsonl"
    backends = list(TARGET_SUITES[target_suite])
    row = {
        "case_index": 0,
        "elapsed_s": 0.1,
        "case": {"case_id": f"case-{name}", "seed": seed, "program": {"operations": []}},
        "findings": findings or [],
    }
    adaptive_components = (run_updates or {}).get("adaptive_components", {})
    adaptive_selection_enabled_by_config = any(
        float((config or {}).get(key, 0.0) or 0.0) > 0.0
        for key in (
            "generator_profile_learning_weight",
            "semantic_objective_learning_weight",
            "metamorphic_relation_learning_weight",
            "version_pair_learning_weight",
            "backend_pair_learning_weight",
        )
    )
    if (
        include_adaptive_selection
        and (
            isinstance(adaptive_components, dict)
            and any(bool(value) for value in adaptive_components.values())
            or adaptive_selection_enabled_by_config
        )
    ):
        row.update(
            {
                "generator_profile_selection": {
                    "strategy": "contextual_bandit",
                    "profile": "common",
                    "profile_pool": ["common"],
                    "learning_weight": 1.0,
                    "reward": 0.5,
                    "ranked": [
                        {
                            "action_id": "common",
                            "score": 0.5,
                            "model_prediction": 0.25,
                            "uncertainty": 0.5,
                            "exploration_bonus": 0.25,
                            "version_signal": 0.1,
                            "continual_priority_signal": 0.1,
                            "health_penalty": 0.0,
                        }
                    ],
                },
                "semantic_objective_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "semantic_objective",
                    "action": "exploration_objective:boundary_depth",
                    "action_pool": ["exploration_objective:boundary_depth"],
                    "learning_weight": 1.0,
                    "reward": 0.5,
                    "ranked": [
                        {
                            "action_id": "exploration_objective:boundary_depth",
                            "uncertainty": 0.5,
                        }
                    ],
                },
                "metamorphic_relation_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "metamorphic_relation",
                    "action": "input_partition_union_all",
                    "action_pool": ["input_partition_union_all"],
                    "learning_weight": 1.0,
                    "reward": 0.5,
                    "ranked": [
                        {
                            "action_id": "input_partition_union_all",
                            "exploration_bonus": 0.25,
                        }
                    ],
                },
                "version_pair_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "version_pair",
                    "action": "latest->fixed",
                    "action_pool": ["latest->fixed"],
                    "learning_weight": 1.0,
                    "reward": 0.5,
                    "ranked": [
                        {
                            "action_id": "latest->fixed",
                            "version_signal": 0.5,
                        }
                    ],
                },
                "backend_pair_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "backend_pair",
                    "priority": ["pandas|duckdb"],
                    "action_pool": ["pandas|duckdb"],
                    "learning_weight": 1.0,
                    "ranked": [
                        {
                            "action_id": "pandas|duckdb",
                            "score": 0.5,
                            "uncertainty": 0.25,
                        }
                    ],
                },
                "backend_pair_priority": ["pandas|duckdb"],
                "selected_generator_profile": "common",
                "selected_semantic_objective": "exploration_objective:boundary_depth",
                "selected_metamorphic_relation": "input_partition_union_all",
                "selected_version_pair": "latest->fixed",
            }
        )
    append_jsonl(row, run_file)
    run_meta = {
        "executed_cases": 1,
        "elapsed_s": 1.0,
        "throughput_cases_s": 1.0,
        "backends": backends,
        "targets": describe_targets(backends),
        "config": config or {},
        "replay_bug_filter": replay_filter or {},
        "run_provenance": run_provenance
        or {
            "schema_version": "run-provenance-v1",
            "vcs": {
                "git_commit": final_readiness.current_workspace_git_commit(),
                "git_commit_short": "current-head",
                "git_branch": "main",
                "workspace_dirty": False,
            },
            "launch": {
                "source": "closed_loop_tmux",
                "session": "authority-test",
                "duration": "24h" if evidence_mode == "live" else "short",
                "batch_duration": "10m",
                "log_prefix": name,
                "launch_script": "start_closed_loop_24h_tmux.sh" if evidence_mode == "live" else "manual",
            },
            "harness": {
                "authority": evidence_mode == "live",
                "freeze_intent": evidence_mode == "live",
                "latest_code_claim": evidence_mode == "live",
                "evidence_role": "latest_live_authority_24h" if evidence_mode == "live" else "",
            },
            "freeze_artifacts": {
                "manifest": str(root / "reports" / f"{name}.freeze.json"),
                "pip_freeze": str(root / "reports" / f"{name}.pip-freeze.txt"),
                "git_status": str(root / "reports" / f"{name}.git-status.txt"),
                "git_diff": str(root / "reports" / f"{name}.git-diff.patch"),
                "launcher_env": str(root / "reports" / f"{name}.launcher-env.txt"),
                "strategy_snapshot": str(root / "reports" / f"{name}.strategy-snapshot.json"),
            },
        },
    }
    if first_candidate_case_index is not None:
        run_meta["first_candidate_bug_case_index"] = first_candidate_case_index
    if first_candidate_elapsed_s is not None:
        run_meta["first_candidate_bug_elapsed_s"] = first_candidate_elapsed_s
    if candidate_bug_discovery_auc is not None:
        run_meta["candidate_bug_discovery_auc"] = candidate_bug_discovery_auc
    value_catalog_enabled = bool(
        not isinstance(adaptive_components, dict)
        or adaptive_components.get("value_catalog", True)
    )
    bd_axis_bandit_enabled = bool(
        not isinstance(adaptive_components, dict)
        or adaptive_components.get("bd_axis_bandit", True)
    )
    seed_energy_tier_enabled = bool(
        not isinstance(adaptive_components, dict)
        or adaptive_components.get("seed_energy_tier", True)
    )
    champion_graft_donor_enabled = bool(
        not isinstance(adaptive_components, dict)
        or adaptive_components.get("champion_graft_donor", True)
    )
    hierarchical_archive_enabled = bool(
        not isinstance(adaptive_components, dict)
        or adaptive_components.get("hierarchical_archive", True)
    )
    run_meta["lhs_seeding"] = {
        "enabled": bool(
            not isinstance(adaptive_components, dict)
            or adaptive_components.get("lhs_seeding", True)
        ),
        "sample_count": 256,
    }
    run_meta["champion_corpus"] = {
        "enabled": bool(
            not isinstance(adaptive_components, dict)
            or adaptive_components.get("champion_corpus", True)
        ),
        "path": str(runs_dir / "champion_corpus.jsonl"),
        "version_id": "test-version",
        "injected_count": 1
        if bool(
            not isinstance(adaptive_components, dict)
            or adaptive_components.get("champion_corpus", True)
        )
        else 0,
    }
    should_write_closed_loop_state = evidence_mode == "live" if include_closed_loop_state is None else include_closed_loop_state
    if should_write_closed_loop_state:
        state_path = runs_dir / f"run-{name}.state.json"
        dump_json({"seen_signatures": [f"sig-{name}"], "feedback": {}, "guidance": {}}, state_path)
        run_meta["closed_loop_state_file"] = str(state_path)
        run_meta["closed_loop_state_summary"] = {
            "seen_signature_count": 1,
            "adaptive_learning_health": {
                "schema_version": "adaptive-learning-health-v1",
                "bandit_count": 1,
                "arm_count": 1,
                "total_pulls": 1,
                "reward_model_update_count": 1,
                "reward_model_feature_count": 2,
                "version_memory_key_count": 1,
                "runtime_cost_observation_count": 1,
                "runtime_cost_total": 0.1,
                "scope_pull_counts": {
                    "bd_axis_weights": 1 if bd_axis_bandit_enabled else 0,
                    "seed_energy_tier": 1 if seed_energy_tier_enabled else 0,
                    "champion_graft_donor": 1 if champion_graft_donor_enabled else 0,
                    "value_catalog_entry": 1 if value_catalog_enabled else 0,
                },
                "scope_arm_counts": {
                    "bd_axis_weights": 1 if bd_axis_bandit_enabled else 0,
                    "seed_energy_tier": 1 if seed_energy_tier_enabled else 0,
                    "champion_graft_donor": 1 if champion_graft_donor_enabled else 0,
                    "value_catalog_entry": 1 if value_catalog_enabled else 0,
                },
                "value_catalog_entry_pulls": 1 if value_catalog_enabled else 0,
                "value_catalog_entry_arm_count": 1 if value_catalog_enabled else 0,
                "bd_axis_weight_pulls": 1 if bd_axis_bandit_enabled else 0,
                "bd_axis_weight_arm_count": 1 if bd_axis_bandit_enabled else 0,
                "seed_energy_tier_pulls": 1 if seed_energy_tier_enabled else 0,
                "seed_energy_tier_arm_count": 1 if seed_energy_tier_enabled else 0,
                "champion_graft_donor_pulls": 1 if champion_graft_donor_enabled else 0,
                "champion_graft_donor_arm_count": 1 if champion_graft_donor_enabled else 0,
                "continual_priority_memory": {
                    "imported_ledger_count": 1,
                    "imported_family_count": 1,
                    "family_count": 1,
                    "feature_count": 2,
                },
                "avg_health_penalty": 0.0,
                "max_health_penalty": 0.0,
                "avg_uncertainty": 0.5,
                "exploration_memory": {
                    "total_records": 1,
                    "context_count": 2,
                    "action_count": 1,
                },
            },
            "quality_archive_health": {
                "schema_version": "quality-archive-health-v1",
                "cell_count": 1,
                "hierarchical_enabled": hierarchical_archive_enabled,
                "child_cell_count": 1 if hierarchical_archive_enabled else 0,
                "split_cell_count": 1 if hierarchical_archive_enabled else 0,
                "seed_count": 1,
                "elite_seed_count": 1,
                "reward_count": 1,
                "outcome_count": 1,
                "invalid_count": 0,
                "fallback_count": 0,
                "false_positive_count": 0,
            },
            "seed_quota_health": {
                "enabled": bool(
                    not isinstance(adaptive_components, dict)
                    or adaptive_components.get("seed_quota", True)
                ),
                "active": bool(
                    not isinstance(adaptive_components, dict)
                    or adaptive_components.get("seed_quota", True)
                ),
                "cluster_count": 1,
                "seed_count": 1,
            },
            "champion_corpus_health": {
                "enabled": bool(
                    not isinstance(adaptive_components, dict)
                    or adaptive_components.get("champion_corpus", True)
                ),
                "family_hit_count": 1
                if bool(
                    not isinstance(adaptive_components, dict)
                    or adaptive_components.get("champion_corpus", True)
                )
                else 0,
                "promoted_family_count": 1
                if bool(
                    not isinstance(adaptive_components, dict)
                    or adaptive_components.get("champion_corpus", True)
                )
                else 0,
                "version_id": "test-version",
            },
        }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    freeze_artifacts = run_meta["run_provenance"].get("freeze_artifacts", {})
    for artifact_name, artifact_path in freeze_artifacts.items():
        path = Path(str(artifact_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{artifact_name}\n", encoding="utf-8")
    if include_stage_profile:
        run_meta["stage_profile"] = {
            "totals_ms": {
                "generate_mutate_ms": 0.1,
                "backend_execution_ms": 0.2,
                "normalize_ms": 0.05,
                "oracle_classification_ms": 0.03,
                "scheduler_feedback_ms": 0.02,
                "logging_artifact_ms": 0.01,
                "total_case_wall_ms": 0.41,
            }
        }
    dump_json(run_meta, run_meta_path(run_file))
    if write_paper_run_journal:
        append_jsonl(
            {
                "run_file": str(run_file),
                "evidence_mode": evidence_mode,
                "target_suite": target_suite,
                "preset": preset,
                "seed": seed,
            },
            reports_dir / "paper-run-journal.jsonl",
        )
    manifest = runs_dir / f"experiment-{name}.json"
    run_payload = {
        "target_suite": target_suite,
        "preset": preset,
        "seed": seed,
        "known_bug_id": known_bug_id,
        "run_file": str(run_file),
        "backends": backends,
        "report": "",
    }
    if run_updates:
        run_payload.update(run_updates)
    manifest_payload = {
        "target_suite": target_suite,
        "target_suites": [target_suite],
        "known_bug_id": known_bug_id,
        "backends": backends,
        "targets": describe_targets(backends),
        "replay_bug_policy": {"enable_replay_bug": evidence_mode == "historical"},
        "runs": [run_payload],
    }
    if evidence_mode is not None:
        manifest_payload["evidence_mode"] = evidence_mode
        run_payload["evidence_mode"] = evidence_mode
    if experiment_meta is not None:
        manifest_payload["experiment_meta"] = experiment_meta
    adaptive_components = run_payload.get("adaptive_components", {})
    if isinstance(adaptive_components, dict) and any(bool(value) for value in adaptive_components.values()):
        manifest_payload["adaptive_state"] = [
            {
                "arm_id": f"{target_suite}:{preset}:0",
                "target_suite": target_suite,
                "preset": preset,
                "pulls": 1,
                "learning_signal": 0.5,
                "bayesian_exploration_observation_count": 1
                if bool(adaptive_components.get("bayesian_exploration", False))
                else 0,
                "annealing_temperature": 0.25
                if bool(adaptive_components.get("scheduler_annealing", False))
                else 0.0,
            }
        ]
    dump_json(manifest_payload, manifest)
    return manifest


def _write_version_ledger(root: Path) -> Path:
    path = root / "reports" / "version-ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        build_version_ledger(
            [
                VersionObservation(
                    version_id="v1",
                    run_file="runs/v1.jsonl",
                    case_count=2,
                    candidate_families={"persistent_family@engine": 1},
                    invalid_case_count=0,
                    fallback_case_count=0,
                    false_positive_count=0,
                    duration_ms_total=4.0,
                    throughput_cases_s=5.0,
                ),
                VersionObservation(
                    version_id="v2",
                    run_file="runs/v2.jsonl",
                    case_count=2,
                    candidate_families={
                        "persistent_family@engine": 1,
                        "new_family@engine": 1,
                    },
                    invalid_case_count=1,
                    fallback_case_count=0,
                    false_positive_count=0,
                    duration_ms_total=5.0,
                    throughput_cases_s=4.0,
                ),
            ]
        ),
        path,
    )
    return path


def _write_target_version_audit(root: Path, *, outdated: bool = False) -> Path:
    path = root / "reports" / ("target-version-audit-outdated.json" if outdated else "target-version-audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    installed = "3.0.3"
    latest = "3.0.4" if outdated else installed
    outdated_rows = (
        [
            {
                "package": "pandas",
                "target": "pandas",
                "installed_version": installed,
                "latest_version": latest,
                "up_to_date": False,
            }
        ]
        if outdated
        else []
    )
    dump_json(
        {
            "schema_version": "target-version-audit-v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "target_packages": [
                {
                    "package": "pandas",
                    "target": "pandas",
                    "role": "implemented target",
                    "installed_version": installed,
                    "latest_version": latest,
                    "up_to_date": not outdated,
                    "latest_source": "override",
                }
            ],
            "summary": {
                "target_package_count": 1,
                "up_to_date_target_package_count": 0 if outdated else 1,
                "outdated_target_package_count": 1 if outdated else 0,
                "unknown_latest_target_package_count": 0,
                "all_target_packages_up_to_date": not outdated,
                "outdated_target_packages": outdated_rows,
                "unknown_latest_target_packages": [],
            },
        },
        path,
    )
    return path
