from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution_accounting import (
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
)
from datadiff.family_novelty import split_family_key
from datadiff.finding_outcomes import candidate_issue_family_keys
from datadiff.goal_first import generate_goal_first_case
from datadiff.metamorphic import all_metamorphic_variants, select_metamorphic_variants
from datadiff.process_cpu_accounting import ProcessCPUAccountingLedger
from datadiff.research_controls.rlcmf_shadow.rlcmf_campaign import OmissionOpportunity, RLCMFCampaignRuntime
from datadiff.research_controls.rlcmf_shadow.rlcmf_counterfactual import (
    CandidateCommonBatch,
    CandidatePolicySelection,
    RLCMFCoupledCounterfactualExecutor,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_factorial import derive_factorial_candidate_row
from datadiff.research_controls.rlcmf_shadow.rlcmf_manifest import validate_rlcmf_manifest
from datadiff.research_controls.rlcmf_shadow.rlcmf_root_capture import RootCaptureLedger
from datadiff.research_controls.rlcmf_shadow.rlcmf_replay import derive_outcome_blind_seeds
from datadiff.research_controls.rlcmf_shadow.rlcmf_statistics import analyze_rlcmf_paired_replay
from datadiff.research_controls.rlcmf_shadow.rlcmf_voi import (
    RootCauseVOIForecast,
    RootCauseVOIForecastSnapshot,
    VOI_FEATURES,
)
from datadiff.runner import run_loaded_case


P7_OUTPUT = Path("experiments/p7_rlcmf_promotion_v1")
P7_AXES = ("candidate", "backend", "relation", "joint")
P7_CASE_COUNT = 64
P7_CASES_PER_MANIFEST = 16
P7_LOW_POOL_SIZE = 2
P7_FULL_POOL_SIZE = 4
P7_LOW_MR_LIMIT = 0
P7_FULL_MR_LIMIT = 2
P7_SEED_PLAN_ID = "p7-rlcmf-powered-paired-seeds-v1"
P7_PREFREEZE_SMOKE_RESERVED_RANGES = (
    (1_470_000_001, 1_470_100_000),
    (1_580_000_001, 1_580_100_000),
    (1_690_000_001, 1_690_100_000),
    (1_800_000_001, 1_800_100_000),
)
P7_CAMPAIGN_SEEDS = derive_outcome_blind_seeds(
    plan_id=P7_SEED_PLAN_ID,
    count=4,
    minimum=0,
    maximum=2147483647,
)

P7_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "ARROW_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}

P7_MANIFEST_SPECS = (
    {
        "spec_id": "core-sql",
        "campaign_seed": P7_CAMPAIGN_SEEDS[0],
        "candidate_seed_start": 1_470_100_001,
        "generator_profile": "p7_core_sql_goal_first",
        "workload": "mixed_projection_filter_aggregate",
        "full_backends": ("pandas", "duckdb", "sqlite", "polars"),
        "low_backends": ("pandas", "duckdb"),
        "mode": "eager",
        "layout": "contiguous",
    },
    {
        "spec_id": "columnar",
        "campaign_seed": P7_CAMPAIGN_SEEDS[1],
        "candidate_seed_start": 1_580_100_001,
        "generator_profile": "p7_columnar_goal_first",
        "workload": "columnar_null_string_numeric",
        "full_backends": ("pandas", "duckdb", "pyarrow", "polars_lazy"),
        "low_backends": ("pandas", "duckdb"),
        "mode": "lazy",
        "layout": "chunked",
    },
    {
        "spec_id": "query-engine",
        "campaign_seed": P7_CAMPAIGN_SEEDS[2],
        "candidate_seed_start": 1_690_100_001,
        "generator_profile": "p7_query_engine_goal_first",
        "workload": "sort_window_union",
        "full_backends": ("pandas", "duckdb", "sqlite", "datafusion"),
        "low_backends": ("pandas", "duckdb"),
        "mode": "query-engine",
        "layout": "contiguous",
    },
    {
        "spec_id": "embedded-sql",
        "campaign_seed": P7_CAMPAIGN_SEEDS[3],
        "candidate_seed_start": 1_800_100_001,
        "generator_profile": "p7_embedded_sql_goal_first",
        "workload": "embedded_sql_boundary",
        "full_backends": ("pandas", "duckdb", "sqlite", "chdb"),
        "low_backends": ("pandas", "duckdb"),
        "mode": "embedded-sql",
        "layout": "contiguous",
    },
)

P7_SOURCE_PATHS = (
    "src/datadiff/research_controls/rlcmf_shadow/p7_rlcmf_promotion.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_counterfactual.py",
    "src/datadiff/risk_limiting_audit.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_sentinel.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_campaign.py",
    "src/datadiff/coverage_debt.py",
    "src/datadiff/process_cpu_accounting.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_root_capture.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_voi.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_safety.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_statistics.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_manifest.py",
    "src/datadiff/run_loaded.py",
    "src/datadiff/runner.py",
    "src/datadiff/goal_first.py",
    "src/datadiff/boundary_values.py",
    "src/datadiff/research_controls/rlcmf_shadow/rlcmf_factorial.py",
    "scripts/run_p7_rlcmf_promotion.py",
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source_manifest(repo_root: Path) -> dict[str, Any]:
    files = []
    for relative in P7_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda row: row["path"])
    return {
        "schema_version": "p7-source-manifest-v1",
        "files": files,
        "source_sha256": canonical_sha256(files),
    }


def _environment_payload() -> dict[str, Any]:
    packages = {}
    for package in (
        "pandas",
        "duckdb",
        "polars",
        "pyarrow",
        "datafusion",
        "chdb",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    return {
        "schema_version": "p7-environment-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key, "") for key in sorted(P7_THREAD_ENV)
        },
        "available_affinity": affinity,
        "selected_affinity_cpu": affinity[0] if affinity else None,
    }


def build_schedule() -> list[dict[str, Any]]:
    schedule = []
    for round_index in range(P7_CASES_PER_MANIFEST):
        for spec_index, spec in enumerate(P7_MANIFEST_SPECS):
            trace_index = len(schedule)
            candidate_seed_start = int(spec["candidate_seed_start"]) + round_index * 100
            row = {
                "trace_index": trace_index,
                "manifest_spec_index": spec_index,
                "spec_id": spec["spec_id"],
                "campaign_seed": spec["campaign_seed"],
                "candidate_seed_start": candidate_seed_start,
                "generator_profile": spec["generator_profile"],
                "workload": spec["workload"],
                "mode": spec["mode"],
                "layout": spec["layout"],
                "low_backends": list(spec["low_backends"]),
                "full_backends": list(spec["full_backends"]),
                "metamorphic_relation_order": [],
            }
            row["case_sha256"] = canonical_sha256(row)
            schedule.append(row)
    if len(schedule) != P7_CASE_COUNT:
        raise AssertionError("P7 schedule size mismatch")
    return schedule


def _manifest_schedule(
    execution_schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "trace_index": int(row["trace_index"]),
            "campaign_seed": int(row["campaign_seed"]),
            "case_sha256": str(row["case_sha256"]),
            "metamorphic_relation_order": list(
                row["metamorphic_relation_order"]
            ),
        }
        for row in execution_schedule
    ]


def _validate_schedule_correspondence(
    manifest_schedule: Sequence[Mapping[str, Any]],
    execution_schedule: Sequence[Mapping[str, Any]],
) -> None:
    if len(manifest_schedule) != len(execution_schedule):
        raise ValueError("P7 manifest and execution schedule lengths differ")
    expected = _manifest_schedule(execution_schedule)
    observed = [dict(row) for row in manifest_schedule]
    if observed != expected:
        raise ValueError(
            "P7 manifest schedule does not correspond to execution schedule"
        )


def _implementation_payload(repo_root: Path) -> dict[str, Any]:
    source = _source_manifest(repo_root)
    environment = _environment_payload()
    design = {
        "method_arm": "p5_promoted_method",
        "axes": list(P7_AXES),
        "case_count": P7_CASE_COUNT,
        "candidate_pool": {"low": P7_LOW_POOL_SIZE, "full": P7_FULL_POOL_SIZE},
        "metamorphic_variant_limit": {
            "low": P7_LOW_MR_LIMIT,
            "full": P7_FULL_MR_LIMIT,
        },
        "manifests": [dict(spec) for spec in P7_MANIFEST_SPECS],
        "coupling": "shared-uniform-comonotone-v1",
        "reuse": "A-adaptive-low_B-low-candidate-full_C-full-static-v1",
        "process_cpu": "resource-self-children-plus-explicit-worker-v1",
        "selection": "frozen-goal-boundary-static-score-v1",
    }
    model = {
        "voi": "registered-linear-root-voi-v1",
        "features": list(VOI_FEATURES),
        "feature_source": "predictable_history_only",
        "cost_metric": "process_cpu_ms",
        "candidate_selection": design["selection"],
    }
    return {
        "source_manifest": source,
        "environment": environment,
        "design": design,
        "model": model,
        "source_sha256": source["source_sha256"],
        "environment_sha256": canonical_sha256(environment),
        "config_sha256": canonical_sha256(design),
        "model_sha256": canonical_sha256(model),
    }


def build_manifest(repo_root: Path) -> dict[str, Any]:
    execution_schedule = build_schedule()
    schedule = _manifest_schedule(execution_schedule)
    trace_sha256 = canonical_sha256(execution_schedule)
    implementation = _implementation_payload(repo_root)
    audit_policy = {
        "floor": 0.05,
        "initial": 0.2,
        "maximum": 1.0,
        "ladder": [0.05, 0.2, 0.5, 1.0],
    }
    threshold = {
        "maximum_event_miss_rate": 0.1,
        "minimum_event_recall": 0.9,
        "maximum_misclassification_rate": 0.05,
        "maximum_uncaptured_full_event_rate": 0.15,
        "caution_margin": 0.02,
    }
    return {
        "schema_version": "rlcmf-manifest-v1",
        "manifest_id": "p7-rlcmf-powered-paired-promotion-v1",
        "outcome_blind": True,
        "seed_plan": {
            "algorithm": "sha256-counter-outcome-blind-v1",
            "plan_id": P7_SEED_PLAN_ID,
            "seeds": [spec["campaign_seed"] for spec in P7_MANIFEST_SPECS],
            "derivation": {
                "serialization": "canonical-json-sort-keys-v1",
                "digest": "sha256-first64-big-endian-v1",
                "minimum": 0,
                "maximum": 2147483647,
                "count": len(P7_MANIFEST_SPECS),
                "collision_rule": "increment-counter-skip-duplicate-v1",
                "schedule": "round-robin-in-trace-order-v1",
            },
            "schedule": schedule,
        },
        "implementation": {
            "source_sha256": implementation["source_sha256"],
            "environment_sha256": implementation["environment_sha256"],
            "config_sha256": implementation["config_sha256"],
            "model_sha256": implementation["model_sha256"],
        },
        "fidelity": {
            "reference_policy": {
                "kind": "live-goal-first-common-batch-v1",
                "backends": sorted(
                    {
                        backend
                        for spec in P7_MANIFEST_SPECS
                        for backend in spec["full_backends"]
                    }
                ),
                "candidate_pool_size": P7_FULL_POOL_SIZE,
                "candidate_recheck_count": 1,
                "metamorphic_variant_limit": P7_FULL_MR_LIMIT,
                "preset": "p5_promoted_method",
                "trace_sha256": trace_sha256,
            },
            "low_policy": {
                "kind": "live-goal-first-common-batch-v1",
                "backend_sample_size": 2,
                "backend_selection": "manifest-fixed-common-controls-v1",
                "candidate_pool_size": P7_LOW_POOL_SIZE,
                "candidate_recheck_count": 0,
                "metamorphic_variant_limit": P7_LOW_MR_LIMIT,
                "preset": "p5_promoted_method",
                "trace_sha256": trace_sha256,
                "voi_forecast_mode": "predictable-history-process-cpu-v1",
            },
            "actions": [
                {"action_id": "candidate-common-batch", "axis": "candidate"},
                {"action_id": "backend-fixed-controls", "axis": "backend"},
                {"action_id": "relation-low-limit", "axis": "relation"},
                {"action_id": "joint-full-static", "axis": "joint"},
            ],
        },
        "audit": {
            "hash_algorithm": "sha256-first64-uniform-v1",
            "serialization": "canonical-json-sort-keys-v1",
            "salt": "p7-rlcmf-powered-paired-promotion-v1:coupled-sentinel",
            "axes": {axis: dict(audit_policy) for axis in P7_AXES},
        },
        "coverage_debt": {
            "forced_settlement": True,
            "strata": [
                {
                    "stratum_id": f"{axis}-fine-grained-wildcard-v1",
                    "axis": axis,
                    "stratum": "*",
                    "max_age_cases": 12,
                    "max_outstanding": 8,
                }
                for axis in P7_AXES
            ],
        },
        "estimators": {
            "event": "doubly_robust",
            "root": "paired_full_reference_only",
            "confidence_sequence": "asymmetric-clipped-freedman-partial-id-v2",
            "alpha": 0.05,
            "alpha_allocation": {axis: 0.0125 for axis in P7_AXES},
            "event_metrics": {
                axis: "any_new_candidate_root" for axis in P7_AXES
            },
            "minimum_effective_audits": 8,
            "weight_clip": 21,
            "event_bound": 1,
        },
        "safety": {
            "states": [
                "calibration_full",
                "adaptive_safe",
                "caution",
                "full_rollback",
            ],
            "decision_rule": "asymmetric-confirmed-violation-v2",
            "rollback_triggers": [
                "non_reproduced_candidate",
                "miss_rate_bound",
                "recall_bound",
                "misclassification_bound",
                "coverage_debt_cap",
                "manifest_integrity",
                "propensity_integrity",
                "ledger_integrity",
                "model_integrity",
            ],
            "propensity_ladder": [0.05, 0.2, 0.5, 1.0],
            "decision_epoch_cases": 8,
            "axis_thresholds": {axis: dict(threshold) for axis in P7_AXES},
            "recovery": {"enabled": False},
            "horizon": {
                "enabled": True,
                "allocation_rule": (
                    "minimum-required-ladder-with-low-path-reserve-v1"
                ),
                "minimum_low_path_cases": 16,
                "debt_reserve_cases": 8,
                "target_confidence_width": 0.2,
                "full_audit_deadline_remaining_cases": 4,
            },
        },
        "voi": {
            "enabled": True,
            "model": "registered-linear-root-voi-v1",
            "model_sha256": implementation["model_sha256"],
            "features": list(VOI_FEATURES),
            "weights": {feature: 1.0 for feature in VOI_FEATURES},
            "minimum_cost_ms": 1.0,
            "raise_thresholds": [
                {"minimum_voi_per_cpu_ms": 0.0005, "ladder_steps": 1},
                {"minimum_voi_per_cpu_ms": 0.005, "ladder_steps": 2},
            ],
            "maximum_extra_expected_cost_ms_per_case": 250.0,
            "raise_joint_with_component": True,
            "permitted_influence": "raise_only",
        },
        "budget": {
            "equal_calibration_cases_per_seed": 1,
            "max_cases_per_seed": P7_CASES_PER_MANIFEST,
            "stopping_rule": (
                "execute all 64 frozen schedule rows; no result-based stop or replacement"
            ),
            "post_outcome_seed_replacement_allowed": False,
            "favorable_early_stopping_allowed": False,
            "paired_analysis": {
                "analysis_id": "p7-powered-paired-cost-analysis-v1",
                "unit": "frozen_trace_case",
                "pairing": "common_case_full_static_shadow",
                "metrics": [
                    "process_cpu_ms",
                    "backend_calls",
                    "backend_reported_ms",
                    "wall_ms",
                ],
                "primary_cost_metric": "process_cpu_ms",
                "confidence_level": 0.95,
                "interval_method": (
                    "circular-moving-block-bootstrap-percentile-v1"
                ),
                "bootstrap_replicates": 10_000,
                "block_length": 8,
                "seed_derivation": (
                    "sha256-manifest-metric-counter-block-v1"
                ),
                "decision_rule": "upper-ratio-ci-below-one-v1",
            },
        },
        "promotion": {
            "primary_metric": (
                "independently_confirmed_unique_real_roots_per_cpu_hour"
            ),
            "requires_all_gates": True,
            "thresholds": {
                "minimum_exact_family_recall": 0.8,
                "minimum_root_recall": 0.9,
                "maximum_non_reproduced_candidate_cases": 0,
                "maximum_misclassification_rate": 0,
                "minimum_primary_metric_ratio": 1.0,
            },
        },
    }


def build_preregistration(repo_root: Path) -> dict[str, Any]:
    execution_schedule = build_schedule()
    manifest_payload = build_manifest(repo_root)
    manifest = validate_rlcmf_manifest(manifest_payload)
    _validate_schedule_correspondence(
        manifest.payload["seed_plan"]["schedule"],
        execution_schedule,
    )
    implementation = _implementation_payload(repo_root)
    return {
        "schema_version": "p7-rlcmf-preregistration-v1",
        "stage": "P7.9",
        "outcome_blind": True,
        "manifest": manifest.payload,
        "manifest_sha256": manifest.sha256,
        "execution_schedule": execution_schedule,
        "execution_schedule_sha256": canonical_sha256(execution_schedule),
        "freshness": {
            "formal_frontiers_executed_before_seal": False,
            "prefreeze_smoke_reserved_seed_ranges": [
                {"minimum": minimum, "maximum": maximum}
                for minimum, maximum in P7_PREFREEZE_SMOKE_RESERVED_RANGES
            ],
            "exclusion_rule": (
                "formal candidate seeds must lie outside every prefreeze smoke "
                "reserved range"
            ),
        },
        "implementation": implementation,
        "primary_hypothesis": (
            "horizon-aware coupled RLCMF reduces process CPU versus full-static "
            "while preserving registered safety evidence"
        ),
        "promotion_rule": {
            "cost": "process_cpu_ms ratio CI upper < 1",
            "safety": "no full rollback, intact controller/propensity/debt logs",
            "root": (
                "identified root evidence or independently confirmed root efficiency; "
                "otherwise remain shadow even if cost improves"
            ),
            "post_outcome_tuning_forbidden": True,
        },
        "negative_result_rule": (
            "remain shadow as an RLCMF safety layer if cost regresses/is "
            "inconclusive or root evidence remains not_identified"
        ),
    }


def prepare(output: Path, *, repo_root: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reserved = [
        output / "preregistration.json",
        output / "manifest.json",
        output / "schedule.json",
        output / "preregistration_seal.json",
    ]
    if any(path.exists() for path in reserved):
        raise FileExistsError("P7 preregistration artifacts already exist")
    preregistration = build_preregistration(repo_root)
    manifest = preregistration["manifest"]
    _write_json(output / "preregistration.json", preregistration)
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "schedule.json",
        {"schedule": preregistration["execution_schedule"]},
    )
    seal = {
        "schema_version": "p7-preregistration-seal-v1",
        "preregistration_sha256": sha256_file(output / "preregistration.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "schedule_sha256": sha256_file(output / "schedule.json"),
        "validated_manifest_sha256": preregistration["manifest_sha256"],
    }
    _write_json(output / "preregistration_seal.json", seal)
    return seal


def load_preregistration(output: Path, *, repo_root: Path) -> dict[str, Any]:
    seal = _read_json(output / "preregistration_seal.json")
    for name, key in (
        ("preregistration.json", "preregistration_sha256"),
        ("manifest.json", "manifest_sha256"),
        ("schedule.json", "schedule_sha256"),
    ):
        if sha256_file(output / name) != seal[key]:
            raise ValueError(f"P7 sealed artifact differs: {name}")
    preregistration = _read_json(output / "preregistration.json")
    manifest = validate_rlcmf_manifest(_read_json(output / "manifest.json"))
    if manifest.sha256 != preregistration["manifest_sha256"]:
        raise ValueError("P7 preregistered manifest digest mismatch")
    execution_schedule = preregistration.get("execution_schedule")
    if not isinstance(execution_schedule, list) or not execution_schedule:
        raise ValueError("P7 preregistration execution schedule is missing")
    if canonical_sha256(execution_schedule) != preregistration.get(
        "execution_schedule_sha256"
    ):
        raise ValueError("P7 preregistered execution schedule digest mismatch")
    schedule_artifact = _read_json(output / "schedule.json").get("schedule")
    if schedule_artifact != execution_schedule:
        raise ValueError("P7 schedule artifact differs from preregistration")
    _validate_schedule_correspondence(
        manifest.payload["seed_plan"]["schedule"],
        execution_schedule,
    )
    observed = _implementation_payload(repo_root)
    expected = preregistration["implementation"]
    for key in (
        "source_sha256",
        "environment_sha256",
        "config_sha256",
        "model_sha256",
    ):
        if observed[key] != expected[key]:
            raise ValueError(f"P7 implementation integrity mismatch: {key}")
    return preregistration


@dataclass(slots=True)
class LiveCandidateBatchProvider:
    frontier_id: str
    seed_start: int
    generator_profile: str
    candidates: list[Case]
    accepted_seeds: list[int]
    skipped_seeds: list[dict[str, Any]]
    seed_cursor: int

    @classmethod
    def create(
        cls,
        *,
        frontier_id: str,
        seed_start: int,
        generator_profile: str,
    ) -> "LiveCandidateBatchProvider":
        return cls(
            frontier_id=frontier_id,
            seed_start=int(seed_start),
            generator_profile=str(generator_profile),
            candidates=[],
            accepted_seeds=[],
            skipped_seeds=[],
            seed_cursor=int(seed_start),
        )

    def ensure(self, count: int) -> CandidateCommonBatch:
        target = int(count)
        attempts = 0
        while len(self.candidates) < target:
            seed = self.seed_cursor
            self.seed_cursor += 1
            attempts += 1
            if attempts > target * 200:
                raise RuntimeError("P7 goal-first generator exhausted bounded attempts")
            generated = generate_goal_first_case(
                seed,
                profile=self.generator_profile,
                boundary_mode="fault_model_targeted",
            )
            if generated.case is None:
                self.skipped_seeds.append(
                    {
                        "seed": seed,
                        "reason": str(generated.trace.get("skip_reason", "")),
                    }
                )
                continue
            self.candidates.append(generated.case)
            self.accepted_seeds.append(seed)
        items = tuple(self.candidates[:target])
        return CandidateCommonBatch.from_items(
            frontier_id=self.frontier_id,
            items=items,
            payload_fn=lambda case: case.to_dict(),
            generation_trace={
                "generator": "goal-first-fault-model-targeted-v1",
                "generator_profile": self.generator_profile,
                "seed_start": self.seed_start,
                "accepted_seeds": self.accepted_seeds[:target],
                "skipped_before_target": [
                    row
                    for row in self.skipped_seeds
                    if row["seed"] < self.accepted_seeds[target - 1]
                ],
            },
        )


def _config(*, metamorphic_limit: int, candidate_recheck_count: int) -> ExperimentConfig:
    return ExperimentConfig(
        method_arm="p5_promoted_method",
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=False,
        enable_metamorphic_oracle=metamorphic_limit > 0,
        metamorphic_variant_limit=int(metamorphic_limit),
        candidate_recheck_count=int(candidate_recheck_count),
        enable_artifact=False,
        enable_feedback=False,
        enable_champion_corpus=False,
        enable_lhs_seeding=False,
        log_level="minimal",
    )


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    return canonical_sha256(snapshot)


def _candidate_score(case: Case) -> tuple[float, dict[str, Any]]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    goal = metadata.get("goal_first_generation", {})
    boundary = metadata.get("boundary_application", {})
    operation_count = len(case.program.operations)
    goal_reached = bool(
        goal.get("goal_reached", goal.get("constructed", metadata.get("goal_id")))
    )
    boundary_applied = bool(
        boundary.get("applied", boundary.get("selected_boundary_id"))
    )
    stable = int(canonical_sha256(case.to_dict())[:16], 16) / float(1 << 64)
    score = (
        float(operation_count)
        + 8.0 * float(goal_reached)
        + 2.0 * float(boundary_applied)
        + stable * 1e-6
    )
    return score, {
        "operation_count": operation_count,
        "goal_reached": goal_reached,
        "boundary_applied": boundary_applied,
        "stable_tie_break": stable,
    }


def _select_candidate(
    items: tuple[Any, ...],
    snapshot: Mapping[str, Any],
    *,
    policy_id: str,
) -> CandidatePolicySelection:
    scored = []
    for index, item in enumerate(items):
        if not isinstance(item, Case):
            raise ValueError("P7 selector requires Case candidates")
        score, trace = _candidate_score(item)
        scored.append((score, -index, index, trace))
    selected = max(scored)
    return CandidatePolicySelection(
        selected_index=selected[2],
        policy_id=policy_id,
        snapshot_sha256=_snapshot_digest(snapshot),
        score=selected[0],
        trace={
            "selection_rule": "goal-boundary-operation-static-score-v1",
            "candidate_scores": [
                {
                    "index": index,
                    "score": score,
                    **trace,
                }
                for score, _negative_index, index, trace in scored
            ],
        },
    )


def _candidate_families(row: Mapping[str, Any]) -> set[str]:
    findings = row.get("findings", [])
    return (
        set(candidate_issue_family_keys(findings))
        if isinstance(findings, list)
        else set()
    )


def _candidate_roots(row: Mapping[str, Any]) -> set[str]:
    return {split_family_key(family)[0] for family in _candidate_families(row)}


def _non_reproduced(row: Mapping[str, Any]) -> bool:
    recheck = row.get("candidate_recheck", {})
    return bool(
        isinstance(recheck, Mapping)
        and recheck.get("non_reproduced_keys", [])
    )


def _voi_snapshot(
    *,
    runtime: RLCMFCampaignRuntime,
    history_rows: Sequence[Mapping[str, Any]],
) -> RootCauseVOIForecastSnapshot:
    history_payload = {
        "completed_cases": runtime.completed_cases,
        "rows": [
            {
                "trace_index": row["trace_index"],
                "adaptive_process_cpu_ms": row["adaptive"][
                    "combined_process_cpu_ms"
                ],
                "reference_process_cpu_ms": row["full_reference_accounting"][
                    "process_cpu_ms"
                ],
                "adaptive_candidate_root_count": len(
                    row["adaptive_candidate_roots"]
                ),
                "reference_candidate_root_count": len(
                    row["full_reference_candidate_roots"]
                ),
            }
            for row in history_rows
        ],
        "safety_decision_id": runtime.current_safety_decision.decision_id,
    }
    history_sha256 = canonical_sha256(history_payload)
    prior_reference_costs = [
        float(row["full_reference_accounting"]["process_cpu_ms"])
        for row in history_rows
    ]
    expected_cost = max(1.0, fmean(prior_reference_costs) if prior_reference_costs else 250.0)
    debt = runtime.coordinator.debt_ledger.summary(
        current_case=runtime.completed_cases
    )
    debt_pressure = min(1.0, float(debt["outstanding_count"]) / 8.0)
    state_uncertainty = 1.0 if runtime.current_safety_decision.state == "caution" else 0.25
    forecasts = []
    for axis, entropy, hazard, diversity in (
        ("candidate", 0.5, 0.4, 0.4),
        ("backend", 0.4, 0.3, 0.5),
        ("relation", 0.3, 0.3, 0.4),
    ):
        forecasts.append(
            RootCauseVOIForecast(
                action_id=f"case-{runtime.completed_cases}:{axis}:root-voi",
                axis=axis,
                history_sha256=history_sha256,
                expected_cost_ms=expected_cost,
                root_entropy_reduction=entropy,
                unseen_root_hazard=hazard,
                disagreement_diversity=diversity,
                coverage_debt_pressure=debt_pressure,
                reproducibility_uncertainty=state_uncertainty,
                saturation_penalty=0.0,
                history_case_count=runtime.completed_cases,
            )
        )
    return RootCauseVOIForecastSnapshot.create(
        manifest_sha256=runtime.manifest.sha256,
        model_sha256=runtime.voi_scheduler.model_sha256,
        history_payload=history_payload,
        history_case_count=runtime.completed_cases,
        forecasts=forecasts,
    )


def _run_reference(
    *,
    schedule_row: Mapping[str, Any],
    policy_snapshot: Mapping[str, Any],
    full_config: ExperimentConfig,
    cost_ledger: ProcessCPUAccountingLedger,
) -> tuple[dict[str, Any], CandidateCommonBatch, CandidatePolicySelection, dict[str, Any]]:
    frontier_id = f"{schedule_row['spec_id']}:seed={schedule_row['candidate_seed_start']}"
    provider = LiveCandidateBatchProvider.create(
        frontier_id=frontier_id,
        seed_start=int(schedule_row["candidate_seed_start"]),
        generator_profile=str(schedule_row["generator_profile"]),
    )
    scope_id = f"reference:{schedule_row['trace_index']}"
    cost_ledger.start_scope(scope_id, metadata={"arm": "full_static_reference"})
    try:
        batch = provider.ensure(P7_FULL_POOL_SIZE)
        selection = _select_candidate(
            batch.items,
            policy_snapshot,
            policy_id="full-static-reference-selection-v1",
        )
        case = batch.items[selection.selected_index]
        row = run_loaded_case(
            case,
            backends=list(schedule_row["full_backends"]),
            config=full_config,
            save_artifact=False,
            config_payload=full_config.to_dict(),
            metamorphic_relation_order=list(
                schedule_row["metamorphic_relation_order"]
            ),
        )
    except Exception:
        cost_ledger.finish_scope(status="error")
        raise
    cost = cost_ledger.finish_scope(
        status=str(row.get("status", "ok") or "ok"),
        backend_reported_ms=execution_profile_backend_reported_ms(
            row.get("execution_profile", {})
        ),
        backend_calls=execution_profile_backend_calls(
            row.get("execution_profile", {})
        ),
    ).to_dict()
    row = dict(row)
    row["p7_process_cost"] = cost
    return row, batch, selection, cost


def run_formal(output: Path, *, repo_root: Path) -> dict[str, Any]:
    if (output / "result.json").exists():
        raise FileExistsError("P7 formal result already exists")
    preregistration = load_preregistration(output, repo_root=repo_root)
    manifest = validate_rlcmf_manifest(preregistration["manifest"])
    schedule = preregistration["execution_schedule"]
    runtime = RLCMFCampaignRuntime(manifest)
    root_ledger = RootCaptureLedger(
        manifest_sha256=manifest.sha256,
        root_clusterer_sha256=canonical_sha256(
            {"algorithm": "split-family-key-root-v1"}
        ),
        design_id="p7-coupled-root-capture-v1",
    )
    case_cost_ledger = ProcessCPUAccountingLedger()
    case_results: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for schedule_row in schedule:
        trace_index = int(schedule_row["trace_index"])
        frontier_id = (
            f"{schedule_row['spec_id']}:seed={schedule_row['candidate_seed_start']}"
        )
        policy_snapshot = {
            "schema_version": "p7-frozen-policy-snapshot-v1",
            "trace_index": trace_index,
            "completed_cases": runtime.completed_cases,
            "safety_decision_id": runtime.current_safety_decision.decision_id,
            "safety_state": runtime.current_safety_decision.state,
            "audit_propensities_exact": {
                axis: str(runtime.current_safety_decision.propensity_for(axis))
                for axis in P7_AXES
            },
            "selection_rule": "goal-boundary-operation-static-score-v1",
            "history_case_digest": canonical_sha256(
                [
                    {
                        "trace_index": row["trace_index"],
                        "adaptive_candidate_roots": row[
                            "adaptive_candidate_roots"
                        ],
                        "adaptive_process_cpu_ms": row["adaptive"][
                            "combined_process_cpu_ms"
                        ],
                    }
                    for row in case_results
                ]
            ),
        }
        low_config = _config(
            metamorphic_limit=P7_LOW_MR_LIMIT,
            candidate_recheck_count=0,
        )
        full_config = _config(
            metamorphic_limit=P7_FULL_MR_LIMIT,
            candidate_recheck_count=1,
        )
        provider = LiveCandidateBatchProvider.create(
            frontier_id=frontier_id,
            seed_start=int(schedule_row["candidate_seed_start"]),
            generator_profile=str(schedule_row["generator_profile"]),
        )
        voi_snapshot = _voi_snapshot(runtime=runtime, history_rows=case_results)
        omissions = {
            "candidate": OmissionOpportunity(
                axis="candidate",
                stratum=(
                    f"generator={schedule_row['generator_profile']}|"
                    f"workload={schedule_row['workload']}|"
                    f"mode={schedule_row['mode']}|layout={schedule_row['layout']}"
                ),
                item_key=frontier_id,
                estimated_omitted_units=P7_FULL_POOL_SIZE - P7_LOW_POOL_SIZE,
            ),
            "backend": OmissionOpportunity(
                axis="backend",
                stratum=f"spec={schedule_row['spec_id']}",
                item_key=(
                    f"case={trace_index}:backend:"
                    + ",".join(
                        sorted(
                            set(schedule_row["full_backends"])
                            - set(schedule_row["low_backends"])
                        )
                    )
                ),
                estimated_omitted_units=(
                    len(schedule_row["full_backends"])
                    - len(schedule_row["low_backends"])
                ),
            ),
            "relation": OmissionOpportunity(
                axis="relation",
                stratum=f"spec={schedule_row['spec_id']}",
                item_key=f"case={trace_index}:relation:full-minus-low",
                estimated_omitted_units=P7_FULL_MR_LIMIT - P7_LOW_MR_LIMIT,
            ),
            "joint": OmissionOpportunity(
                axis="joint",
                stratum=f"spec={schedule_row['spec_id']}",
                item_key=f"case={trace_index}:joint",
                estimated_omitted_units=(
                    len(schedule_row["full_backends"])
                    * (1 + P7_FULL_MR_LIMIT)
                ),
            ),
        }

        adaptive_scope = f"adaptive:{trace_index}"
        case_cost_ledger.start_scope(
            adaptive_scope,
            metadata={"trace_index": trace_index, "arm": "adaptive"},
        )
        try:
            plans = runtime.plan_case(
                campaign_seed=int(schedule_row["campaign_seed"]),
                omissions=omissions,
                voi_snapshot=voi_snapshot,
                coupled=True,
                coupling_key=f"trace={trace_index}:all-axes",
            )
            decision_set = runtime.coordinator.coupled_decision_sets[
                plans["joint"].coupling_id
            ]
            group = root_ledger.register_group(
                opportunity_id=f"trace={trace_index}:full-outcome-union",
                axis_arms=P7_AXES,
                inclusion_propensity=max(
                    plan.decision.propensity for plan in plans.values()
                ),
                selected=any(plan.selected for plan in plans.values()),
                independence_block_id=plans["joint"].coupling_id,
                coupling_id=plans["joint"].coupling_id,
                metadata={
                    "trace_index": trace_index,
                    "coupled_decision_set": decision_set.to_dict(),
                },
            )
            inner_cost_ledger = ProcessCPUAccountingLedger()

            def run_pipeline(**kwargs):
                backend_fidelity = kwargs["backend_fidelity"]
                relation_fidelity = kwargs["relation_fidelity"]
                config = (
                    full_config
                    if relation_fidelity == "full"
                    else low_config
                )
                backends = (
                    list(schedule_row["full_backends"])
                    if backend_fidelity == "full"
                    else list(schedule_row["low_backends"])
                )
                return run_loaded_case(
                    kwargs["candidate"],
                    backends=backends,
                    config=config,
                    save_artifact=False,
                    config_payload=config.to_dict(),
                    metamorphic_relation_order=list(
                        schedule_row["metamorphic_relation_order"]
                    ),
                )

            def derive_factorial(**kwargs):
                case = kwargs["candidate"]
                variants = list(all_metamorphic_variants(case))
                axis = kwargs["axis"]
                config = low_config if axis == "backend" else full_config
                selected_variants = select_metamorphic_variants(
                    variants,
                    limit=(P7_LOW_MR_LIMIT if axis == "backend" else P7_FULL_MR_LIMIT),
                    relation_order=list(
                        schedule_row["metamorphic_relation_order"]
                    ),
                )
                return derive_factorial_candidate_row(
                    case,
                    kwargs["full_pipeline_row"],
                    backends=(
                        list(schedule_row["full_backends"])
                        if axis == "backend"
                        else list(schedule_row["low_backends"])
                    ),
                    variant_names=[variant.name for variant in selected_variants],
                    config=config,
                )

            executor = RLCMFCoupledCounterfactualExecutor(
                coordinator=runtime.coordinator,
                generate_low_batch_fn=lambda size: provider.ensure(size),
                generate_full_batch_fn=lambda size: provider.ensure(size),
                select_low_fn=lambda items, snapshot: _select_candidate(
                    items,
                    snapshot,
                    policy_id="p7-reduced-pool-selection-v1",
                ),
                select_full_fn=lambda items, snapshot: _select_candidate(
                    items,
                    snapshot,
                    policy_id="p7-full-pool-selection-v1",
                ),
                run_pipeline_fn=run_pipeline,
                factorial_deriver=derive_factorial,
                process_cpu_ledger=inner_cost_ledger,
                require_process_cpu_accounting=True,
            )
            coupled_result = executor.execute(
                plans=plans,
                frontier_id=frontier_id,
                low_pool_size=P7_LOW_POOL_SIZE,
                full_pool_size=P7_FULL_POOL_SIZE,
                policy_snapshot=policy_snapshot,
            )
            adaptive_candidate_roots = sorted(
                {
                    root
                    for row in coupled_result.rows.values()
                    for root in _candidate_roots(row)
                }
            )
            adaptive_candidate_families = sorted(
                {
                    family
                    for row in coupled_result.rows.values()
                    for family in _candidate_families(row)
                }
            )
            captured_roots = sorted(
                {
                    root
                    for comparison in coupled_result.comparisons.values()
                    for root in comparison.new_candidate_roots
                }
            )
            for root in captured_roots:
                family_keys = sorted(
                    family
                    for family in adaptive_candidate_families
                    if split_family_key(family)[0] == root
                ) or [root]
                root_ledger.record_capture(
                    root_id=root,
                    group_id=group.group_id,
                    evidence_level="candidate_root",
                    candidate_family_keys=family_keys,
                    raw_to_root_mapping={family: root for family in family_keys},
                    metadata={"trace_index": trace_index},
                )
            non_reproduced = any(
                _non_reproduced(row) for row in coupled_result.rows.values()
            )
            confirmed_misclassification = any(
                comparison.low_only_confirmed_roots
                for comparison in coupled_result.comparisons.values()
            )
            runtime.complete_case(
                non_reproduced_candidate_increment=int(non_reproduced),
                confirmed_misclassification_violation=bool(
                    confirmed_misclassification
                ),
            )
        except Exception:
            case_cost_ledger.finish_scope(status="error")
            raise
        coupled_summary = coupled_result.summary()
        adaptive_cost = case_cost_ledger.finish_scope(
            status="ok",
            backend_reported_ms=float(coupled_summary["backend_reported_ms"]),
            backend_calls=int(coupled_summary["backend_calls"]),
        ).to_dict()

        reference_row, reference_batch, reference_selection, reference_cost = (
            _run_reference(
                schedule_row=schedule_row,
                policy_snapshot=policy_snapshot,
                full_config=full_config,
                cost_ledger=case_cost_ledger,
            )
        )
        if coupled_result.common_batch is not None:
            common_batch_match = (
                coupled_result.common_batch.batch_sha256
                == reference_batch.batch_sha256
            )
        else:
            common_batch_match = (
                coupled_result.low_batch.candidate_sha256s
                == reference_batch.candidate_sha256s[:P7_LOW_POOL_SIZE]
            )
        if not common_batch_match:
            raise ValueError("P7 adaptive/reference candidate frontier mismatch")

        full_reference_families = sorted(_candidate_families(reference_row))
        full_reference_roots = sorted(_candidate_roots(reference_row))
        case_row = {
            "trace_index": trace_index,
            "schedule": dict(schedule_row),
            "frontier_id": frontier_id,
            "policy_snapshot": policy_snapshot,
            "audit_plans": {
                axis: plan.to_dict() for axis, plan in sorted(plans.items())
            },
            "coupled_decision_set": decision_set.to_dict(),
            "adaptive": {
                **coupled_summary,
                "combined_backend_calls": int(coupled_summary["backend_calls"]),
                "combined_backend_reported_ms": float(
                    coupled_summary["backend_reported_ms"]
                ),
                "combined_wall_ms": float(adaptive_cost["wall_ms"]),
                "combined_process_cpu_ms": float(
                    adaptive_cost["total_process_cpu_ms"]
                ),
                "outer_process_cost": adaptive_cost,
                "inner_process_cost": inner_cost_ledger.summary(),
            },
            "adaptive_candidate_families": adaptive_candidate_families,
            "adaptive_candidate_roots": adaptive_candidate_roots,
            "full_reference_accounting": {
                "backend_calls": int(reference_cost["backend_calls"]),
                "backend_reported_ms": float(
                    reference_cost["backend_reported_ms"]
                ),
                "wall_ms": float(reference_cost["wall_ms"]),
                "process_cpu_ms": float(reference_cost["total_process_cpu_ms"]),
            },
            "full_reference_candidate_families": full_reference_families,
            "full_reference_candidate_roots": full_reference_roots,
            "full_reference_status": reference_row.get("status"),
            "full_reference_selection": {
                "selected_index": reference_selection.selected_index,
                "candidate_sha256": reference_batch.candidate_sha256s[
                    reference_selection.selected_index
                ],
                "batch_sha256": reference_batch.batch_sha256,
                "score": reference_selection.score,
                "trace": dict(reference_selection.trace),
            },
            "candidate_frontier_match": common_batch_match,
            "safety_state_after": runtime.current_safety_decision.state,
            "safety_decision_after": runtime.current_safety_decision.to_dict(),
            "root_capture_group_id": group.group_id,
        }
        case_results.append(case_row)
        trace_rows.append(
            {
                "trace_index": trace_index,
                "frontier_id": frontier_id,
                "adaptive_process_cpu_ms": case_row["adaptive"][
                    "combined_process_cpu_ms"
                ],
                "reference_process_cpu_ms": case_row[
                    "full_reference_accounting"
                ]["process_cpu_ms"],
                "selected_axes": sorted(
                    axis for axis, plan in plans.items() if plan.selected
                ),
                "safety_state_after": runtime.current_safety_decision.state,
            }
        )

    adaptive_families = sorted(
        {
            family
            for row in case_results
            for family in row["adaptive_candidate_families"]
        }
    )
    reference_families = sorted(
        {
            family
            for row in case_results
            for family in row["full_reference_candidate_families"]
        }
    )
    adaptive_roots = sorted(
        {
            root
            for row in case_results
            for root in row["adaptive_candidate_roots"]
        }
    )
    reference_roots = sorted(
        {
            root
            for row in case_results
            for root in row["full_reference_candidate_roots"]
        }
    )
    result = {
        "schema_version": "p7-rlcmf-powered-paired-result-v1",
        "manifest_id": manifest.payload["manifest_id"],
        "manifest_sha256": manifest.sha256,
        "status": "complete",
        "measurement_scope": (
            "fresh_goal_first_paired_process_cpu_candidate_evidence_not_confirmed_real_bugs"
        ),
        "trace": {
            "case_count": len(case_results),
            "trace_sha256": manifest.payload["fidelity"]["reference_policy"][
                "trace_sha256"
            ],
            "result_trace_sha256": canonical_sha256(trace_rows),
        },
        "metrics": {
            "adaptive_backend_calls": sum(
                row["adaptive"]["combined_backend_calls"] for row in case_results
            ),
            "full_reference_backend_calls": sum(
                row["full_reference_accounting"]["backend_calls"]
                for row in case_results
            ),
            "adaptive_process_cpu_ms": sum(
                row["adaptive"]["combined_process_cpu_ms"]
                for row in case_results
            ),
            "full_reference_process_cpu_ms": sum(
                row["full_reference_accounting"]["process_cpu_ms"]
                for row in case_results
            ),
            "adaptive_wall_ms": sum(
                row["adaptive"]["combined_wall_ms"] for row in case_results
            ),
            "full_reference_wall_ms": sum(
                row["full_reference_accounting"]["wall_ms"]
                for row in case_results
            ),
            "adaptive_backend_reported_ms": sum(
                row["adaptive"]["combined_backend_reported_ms"]
                for row in case_results
            ),
            "full_reference_backend_reported_ms": sum(
                row["full_reference_accounting"]["backend_reported_ms"]
                for row in case_results
            ),
            "adaptive_candidate_families": adaptive_families,
            "full_reference_candidate_families": reference_families,
            "adaptive_candidate_roots": adaptive_roots,
            "full_reference_candidate_roots": reference_roots,
            "candidate_family_recall": (
                len(set(adaptive_families) & set(reference_families))
                / len(reference_families)
                if reference_families
                else None
            ),
            "candidate_root_recall": (
                len(set(adaptive_roots) & set(reference_roots))
                / len(reference_roots)
                if reference_roots
                else None
            ),
            "independently_confirmed_unique_real_roots_per_cpu_hour": None,
        },
        "safety": {
            "final_state": runtime.current_safety_decision.state,
            "controller_log_integrity": runtime.controller.verify_log(),
            "pending_sentinel_plans": runtime.coordinator.to_dict()[
                "pending_plan_ids"
            ],
            "outstanding_debt": runtime.coordinator.debt_ledger.summary(
                current_case=runtime.completed_cases
            ),
        },
        "root_capture": root_ledger.to_dict(),
        "campaign": runtime.to_dict(),
        "case_process_costs": case_cost_ledger.summary(),
        "case_results": case_results,
    }
    analysis = analyze_rlcmf_paired_replay(manifest, result)
    process_cpu = analysis["metrics"]["process_cpu_ms"]
    root_readiness = result["root_capture"]["readiness"]
    gates = {
        "process_cpu_optimized": process_cpu["ratio_confidence_interval"][
            "upper"
        ]
        < 1.0,
        "safety_no_full_rollback": result["safety"]["final_state"]
        != "full_rollback",
        "controller_log_integrity": result["safety"][
            "controller_log_integrity"
        ],
        "no_pending_sentinel_plans": not result["safety"][
            "pending_sentinel_plans"
        ],
        "candidate_frontier_integrity": all(
            row["candidate_frontier_match"] for row in case_results
        ),
        "root_evidence_identified": bool(
            root_readiness["primary_root_efficiency_ready"]
        ),
    }
    promotion = {
        "schema_version": "p7-rlcmf-promotion-decision-v1",
        "gates": gates,
        "eligible": all(gates.values()),
        "decision": "promote" if all(gates.values()) else "remain_shadow",
        "reason": (
            "all preregistered cost, safety, integrity, and root-evidence gates passed"
            if all(gates.values())
            else "at least one preregistered gate failed or remained not_identified"
        ),
        "claim_boundary": (
            "candidate roots are not independently confirmed real roots; process CPU "
            "optimization alone does not override not_identified root evidence"
        ),
    }
    result["promotion"] = promotion
    analysis["promotion"] = promotion
    _write_json(output / "result.json", result)
    _write_json(output / "analysis.json", analysis)
    _write_json(output / "root_capture.json", result["root_capture"])
    _write_json(output / "campaign_state.json", result["campaign"])
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        render_report(result, analysis),
        encoding="utf-8",
    )
    return {"result": result, "analysis": analysis}


def render_report(result: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    cpu = analysis["metrics"]["process_cpu_ms"]
    calls = analysis["metrics"]["backend_calls"]
    wall = analysis["metrics"]["wall_ms"]
    promotion = result["promotion"]
    return f"""# P7 RLCMF powered paired promotion experiment

The 64-case schedule, four manifest blocks, coupled draw, horizon controller,
process-CPU primary metric, bootstrap, and gates were sealed before execution.

## Cost

- Process CPU ratio: {cpu['ratio_of_totals']:.6f}, 95% moving-block CI
  [{cpu['ratio_confidence_interval']['lower']:.6f},
  {cpu['ratio_confidence_interval']['upper']:.6f}].
- Backend-call ratio: {calls['ratio_of_totals']:.6f}, CI
  [{calls['ratio_confidence_interval']['lower']:.6f},
  {calls['ratio_confidence_interval']['upper']:.6f}].
- Wall ratio: {wall['ratio_of_totals']:.6f}, CI
  [{wall['ratio_confidence_interval']['lower']:.6f},
  {wall['ratio_confidence_interval']['upper']:.6f}].

Wall, backend-reported duration, and process CPU remain separate measurements.

## Safety and root evidence

- Final controller state: `{result['safety']['final_state']}`.
- Controller log integrity: `{result['safety']['controller_log_integrity']}`.
- Adaptive/reference candidate roots:
  {len(result['metrics']['adaptive_candidate_roots'])}/
  {len(result['metrics']['full_reference_candidate_roots'])}.
- Independently confirmed roots/CPU-hour: unavailable.
- Root efficiency readiness:
  `{result['root_capture']['readiness']['primary_root_efficiency_ready']}`.

## Decision

`{promotion['decision']}`. {promotion['reason']}

{promotion['claim_boundary']}
"""


def finalize(output: Path, *, repo_root: Path) -> dict[str, Any]:
    load_preregistration(output, repo_root=repo_root)
    if not (output / "result.json").is_file():
        raise FileNotFoundError(output / "result.json")
    frozen_root = output / "frozen_source"
    if frozen_root.exists():
        raise FileExistsError("P7 frozen source archive already exists")
    source_manifest = _source_manifest(repo_root)
    for row in source_manifest["files"]:
        destination = frozen_root / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / row["path"], destination)
    archive_rows = []
    for path in sorted(frozen_root.rglob("*")):
        if path.is_file():
            archive_rows.append(
                {
                    "path": path.relative_to(frozen_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    archive_manifest = {
        "schema_version": "p7-frozen-source-archive-v1",
        "source_sha256": canonical_sha256(archive_rows),
        "files": archive_rows,
    }
    _write_json(frozen_root / "archive_manifest.json", archive_manifest)
    checksum_path = output / "checksums.sha256"
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(repo_root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": "p7-finalization-v1",
        "frozen_source_sha256": archive_manifest["source_sha256"],
        "frozen_source_file_count": len(archive_rows),
        "checksum_entry_count": len(paths),
    }


def verify(output: Path, *, repo_root: Path) -> dict[str, Any]:
    preregistration = load_preregistration(output, repo_root=repo_root)
    manifest = validate_rlcmf_manifest(preregistration["manifest"])
    result = _read_json(output / "result.json")
    analysis = _read_json(output / "analysis.json")
    if result["manifest_sha256"] != manifest.sha256:
        raise ValueError("P7 result manifest digest mismatch")
    recomputed = analyze_rlcmf_paired_replay(manifest, result)
    recomputed["promotion"] = analysis["promotion"]
    if recomputed != analysis:
        raise ValueError("P7 analysis is not reproducible")
    checksums = {}
    for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    for relative, digest in checksums.items():
        if sha256_file(repo_root / relative) != digest:
            raise ValueError(f"P7 checksum mismatch: {relative}")
    return {
        "schema_version": "p7-verification-v1",
        "case_count": len(result["case_results"]),
        "manifest_sha256": manifest.sha256,
        "result_sha256": sha256_file(output / "result.json"),
        "analysis_sha256": sha256_file(output / "analysis.json"),
        "promotion_decision": result["promotion"]["decision"],
        "all_pass": True,
    }
