from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from datadiff.backend_sampling import CoverageAwareBackendSampler
from datadiff.candidate_burst import candidate_family_evidence
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution_accounting import (
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
)
from datadiff.family_novelty import split_family_key
from datadiff.finding_outcomes import candidate_issue_family_keys
from datadiff.metamorphic import (
    all_metamorphic_variants,
    select_metamorphic_variants,
)
from datadiff.preset_catalog import build_catalog_preset
from datadiff.research_controls.rlcmf_shadow.rlcmf_campaign import OmissionOpportunity, RLCMFCampaignRuntime
from datadiff.research_controls.rlcmf_shadow.rlcmf_counterfactual import CounterfactualCandidateComparison
from datadiff.research_controls.rlcmf_shadow.rlcmf_manifest import ValidatedRLCMFManifest
from datadiff.research_controls.rlcmf_shadow.rlcmf_runner import run_rlcmf_audited_loaded_case


@dataclass(frozen=True, slots=True)
class RLCMFReplayPolicy:
    preset: str
    backends: tuple[str, ...]
    low_backend_sample_size: int
    low_metamorphic_variant_limit: int
    reference_metamorphic_variant_limit: int
    low_candidate_recheck_count: int
    reference_candidate_recheck_count: int
    voi_forecast_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "backends": list(self.backends),
            "low_backend_sample_size": self.low_backend_sample_size,
            "low_metamorphic_variant_limit": self.low_metamorphic_variant_limit,
            "reference_metamorphic_variant_limit": (
                self.reference_metamorphic_variant_limit
            ),
            "low_candidate_recheck_count": self.low_candidate_recheck_count,
            "reference_candidate_recheck_count": (
                self.reference_candidate_recheck_count
            ),
            "backend_selection": "coverage-balanced-v1",
            "voi_forecast_mode": self.voi_forecast_mode,
        }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def derive_outcome_blind_seeds(
    *,
    plan_id: str,
    count: int,
    minimum: int = 0,
    maximum: int = (2**31) - 1,
) -> list[int]:
    normalized_plan_id = str(plan_id or "").strip()
    if not normalized_plan_id:
        raise ValueError("seed plan_id cannot be empty")
    requested = int(count)
    lower = int(minimum)
    upper = int(maximum)
    if requested <= 0:
        raise ValueError("seed count must be positive")
    if lower < 0 or upper < lower:
        raise ValueError("seed bounds must satisfy 0 <= minimum <= maximum")
    span = upper - lower + 1
    if requested > span:
        raise ValueError("seed range is too small for the requested unique count")
    seeds: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(seeds) < requested:
        material = {
            "algorithm": "sha256-counter-outcome-blind-v1",
            "plan_id": normalized_plan_id,
            "counter": counter,
        }
        draw = int.from_bytes(
            hashlib.sha256(_canonical_json_bytes(material)).digest()[:8],
            "big",
        )
        seed = lower + (draw % span)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
        counter += 1
    return seeds


def build_frozen_round_robin_schedule(
    trace_rows: list[Mapping[str, Any]],
    *,
    campaign_seeds: list[int] | tuple[int, ...],
    metamorphic_relation_order: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    seeds = tuple(int(seed) for seed in campaign_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("campaign seeds must be unique and non-negative")
    relations = tuple(str(value).strip() for value in metamorphic_relation_order)
    if any(not value for value in relations) or len(set(relations)) != len(relations):
        raise ValueError("metamorphic relation order must be unique and non-empty")
    schedule: list[dict[str, Any]] = []
    for index, row in enumerate(trace_rows):
        case_payload = row.get("case")
        if not isinstance(case_payload, Mapping):
            raise ValueError(f"trace row {index} lacks a case payload")
        schedule.append(
            {
                "trace_index": index,
                "campaign_seed": seeds[index % len(seeds)],
                "case_sha256": canonical_payload_sha256(case_payload),
                "metamorphic_relation_order": list(relations),
            }
        )
    return schedule


def parse_replay_policy(manifest: ValidatedRLCMFManifest) -> RLCMFReplayPolicy:
    registered_axes = set(manifest.payload.get("audit", {}).get("axes", {}))
    if registered_axes != {"backend", "relation", "joint"}:
        raise ValueError(
            "paired replay v1 supports exactly backend, relation, and joint axes"
        )
    fidelity = manifest.payload["fidelity"]
    reference = fidelity["reference_policy"]
    low = fidelity["low_policy"]
    if reference.get("kind") != "paired-replay-v1" or low.get("kind") != "paired-replay-v1":
        raise ValueError("paired replay requires paired-replay-v1 fidelity policies")
    if low.get("backend_selection") != "coverage-balanced-v1":
        raise ValueError("unsupported paired-replay backend selection")
    if low.get("voi_forecast_mode") != "none-v1":
        raise ValueError(
            "paired replay v1 requires the frozen outcome-blind none-v1 VOI forecast mode"
        )
    preset = str(reference.get("preset", "")).strip()
    if not preset or str(low.get("preset", "")).strip() != preset:
        raise ValueError("low and reference replay policies must share one preset")
    raw_backends = reference.get("backends")
    if not isinstance(raw_backends, list):
        raise ValueError("reference replay backends must be a list")
    backends = tuple(str(value).strip() for value in raw_backends)
    if len(backends) < 2 or any(not value for value in backends):
        raise ValueError("paired replay requires at least two named backends")
    if len(set(backends)) != len(backends):
        raise ValueError("paired replay backends must be unique")
    sample_size = int(low.get("backend_sample_size", 0))
    if not 2 <= sample_size < len(backends):
        raise ValueError("low backend sample size must omit at least one backend")
    low_mr = int(low.get("metamorphic_variant_limit", -1))
    reference_mr = int(reference.get("metamorphic_variant_limit", -1))
    if low_mr < 0 or reference_mr <= low_mr:
        raise ValueError("reference metamorphic limit must exceed the low limit")
    low_recheck = int(low.get("candidate_recheck_count", -1))
    reference_recheck = int(reference.get("candidate_recheck_count", -1))
    if low_recheck < 0 or reference_recheck < low_recheck:
        raise ValueError("reference candidate rechecks cannot be below the low policy")
    return RLCMFReplayPolicy(
        preset=preset,
        backends=backends,
        low_backend_sample_size=sample_size,
        low_metamorphic_variant_limit=low_mr,
        reference_metamorphic_variant_limit=reference_mr,
        low_candidate_recheck_count=low_recheck,
        reference_candidate_recheck_count=reference_recheck,
        voi_forecast_mode="none-v1",
    )


def validate_replay_trace(
    manifest: ValidatedRLCMFManifest,
    trace_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    schedule = manifest.payload["seed_plan"].get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("paired replay requires a frozen seed_plan.schedule")
    if len(schedule) != len(trace_rows):
        raise ValueError("frozen replay schedule and case trace lengths differ")
    normalized: list[dict[str, Any]] = []
    for index, (entry, trace_row) in enumerate(zip(schedule, trace_rows, strict=True)):
        case_payload = trace_row.get("case")
        if not isinstance(case_payload, Mapping):
            raise ValueError(f"trace row {index} lacks a case payload")
        actual_hash = canonical_payload_sha256(case_payload)
        if actual_hash != entry["case_sha256"]:
            raise ValueError(f"frozen case digest mismatch at trace index {index}")
        normalized.append(dict(entry))
    return normalized


def _config_for_policy(
    policy: RLCMFReplayPolicy,
    *,
    reference: bool,
) -> ExperimentConfig:
    config = build_catalog_preset(policy.preset)
    if config is None:
        raise ValueError(f"unknown paired replay preset: {policy.preset}")
    config.enable_backend_sampling = False
    config.enable_artifact = False
    config.enable_reducer = False
    config.metamorphic_variant_limit = (
        policy.reference_metamorphic_variant_limit
        if reference
        else policy.low_metamorphic_variant_limit
    )
    config.candidate_recheck_count = (
        policy.reference_candidate_recheck_count
        if reference
        else policy.low_candidate_recheck_count
    )
    return config


def _roots(families: Iterable[str]) -> set[str]:
    return {split_family_key(family)[0] for family in families}


def _observed_families(row: Mapping[str, Any]) -> set[str]:
    findings = row.get("findings", []) or []
    return set(candidate_issue_family_keys(findings if isinstance(findings, list) else []))


def _confirmed_families(row: Mapping[str, Any]) -> set[str]:
    evidence = candidate_family_evidence(dict(row))
    return set(str(value) for value in evidence["confirmed_candidate_families"])


def _has_non_reproduced_candidate(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        bool(
            (row.get("candidate_recheck", {}) or {}).get(
                "non_reproduced_keys", []
            )
        )
        for row in rows
    )


def _recall(discovered: set[str], reference: set[str]) -> float | None:
    if not reference:
        return None
    return len(discovered & reference) / len(reference)


def run_rlcmf_paired_replay(
    manifest: ValidatedRLCMFManifest,
    trace_rows: list[Mapping[str, Any]],
    *,
    run_loaded_case_fn: Callable[..., dict[str, Any]],
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
    backend_instances: dict[str, Any] | None = None,
    execution_session: Any | None = None,
) -> dict[str, Any]:
    policy = parse_replay_policy(manifest)
    schedule = validate_replay_trace(manifest, trace_rows)
    cases = [Case.from_dict(dict(row["case"])) for row in trace_rows]
    low_config = _config_for_policy(policy, reference=False)
    reference_config = _config_for_policy(policy, reference=True)
    runtime = RLCMFCampaignRuntime(manifest)
    sampler = CoverageAwareBackendSampler(
        policy.backends,
        enabled=True,
        sample_size=policy.low_backend_sample_size,
        full_sweep_interval=0,
        calibration_cases=0,
        candidate_burst_cases=0,
    )
    resolved_targets = list(target_specs or [])
    adaptive_families: set[str] = set()
    adaptive_confirmed_families: set[str] = set()
    reference_families: set[str] = set()
    reference_confirmed_families: set[str] = set()
    adaptive_wall_ms = 0.0
    adaptive_backend_ms = 0.0
    adaptive_backend_calls = 0
    reference_wall_ms = 0.0
    reference_backend_ms = 0.0
    reference_backend_calls = 0
    validation_extra_wall_ms = 0.0
    validation_extra_backend_ms = 0.0
    validation_extra_backend_calls = 0
    case_results: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    for index, (case, schedule_entry) in enumerate(
        zip(cases, schedule, strict=True)
    ):
        relation_order = list(schedule_entry["metamorphic_relation_order"])
        applicable = list(all_metamorphic_variants(case))
        low_variants = select_metamorphic_variants(
            applicable,
            limit=policy.low_metamorphic_variant_limit,
            relation_order=relation_order,
        )
        reference_variants = select_metamorphic_variants(
            applicable,
            limit=policy.reference_metamorphic_variant_limit,
            relation_order=relation_order,
        )
        low_variant_names = {variant.name for variant in low_variants}
        omitted_variants = [
            variant
            for variant in reference_variants
            if variant.name not in low_variant_names
        ]
        backend_selection = sampler.select()
        low_backends = list(backend_selection.active_backends)
        omitted_backends = list(backend_selection.omitted_backends)
        case_hash = str(schedule_entry["case_sha256"])
        campaign_seed = int(schedule_entry["campaign_seed"])
        stratum = f"seed={campaign_seed}"
        omissions: dict[str, OmissionOpportunity] = {}
        if omitted_backends:
            omissions["backend"] = OmissionOpportunity(
                axis="backend",
                stratum=stratum,
                item_key=(
                    f"{case_hash}:backend:{','.join(sorted(omitted_backends))}"
                ),
                estimated_omitted_units=max(
                    1,
                    len(omitted_backends) * (1 + len(low_variants)),
                ),
            )
        if omitted_variants:
            omissions["relation"] = OmissionOpportunity(
                axis="relation",
                stratum=stratum,
                item_key=(
                    f"{case_hash}:relation:"
                    + ",".join(sorted(variant.name for variant in omitted_variants))
                ),
                estimated_omitted_units=max(
                    1,
                    len(omitted_variants) * len(low_backends),
                ),
            )
        if omissions:
            low_calls = len(low_backends) * (1 + len(low_variants))
            full_calls = len(policy.backends) * (1 + len(reference_variants))
            omissions["joint"] = OmissionOpportunity(
                axis="joint",
                stratum=stratum,
                item_key=f"{case_hash}:joint",
                estimated_omitted_units=max(1, full_calls - low_calls),
            )
        safety_before = runtime.current_safety_decision
        plans = runtime.plan_case(
            campaign_seed=campaign_seed,
            omissions=omissions,
            voi_forecasts=(),
        )
        audited = run_rlcmf_audited_loaded_case(
            case,
            coordinator=runtime.coordinator,
            plans=plans,
            low_backends=low_backends,
            reference_backends=list(policy.backends),
            low_config=low_config,
            reference_config=reference_config,
            low_metamorphic_relation_order=relation_order,
            reference_metamorphic_relation_order=relation_order,
            backend_instances=backend_instances,
            execution_session=execution_session,
            environment=environment,
            target_specs=resolved_targets,
            save_low_artifact=False,
            save_audit_artifacts=False,
            run_loaded_case_fn=run_loaded_case_fn,
        )
        adaptive_summary = audited.summary()
        adaptive_wall_ms += float(adaptive_summary["combined_wall_ms"])
        adaptive_backend_ms += float(
            adaptive_summary["combined_backend_reported_ms"]
        )
        adaptive_backend_calls += int(adaptive_summary["combined_backend_calls"])
        if "joint" in audited.audit_rows:
            full_reference_row = audited.audit_rows["joint"]
            full_reference_reused = True
        else:
            full_reference_row = run_loaded_case_fn(
                case,
                backends=list(policy.backends),
                config=reference_config,
                save_artifact=False,
                backend_instances=None,
                environment=environment,
                target_specs=resolved_targets,
                config_payload=reference_config.to_dict(),
                metamorphic_relation_order=relation_order,
                execution_session=None,
            )
            full_reference_reused = False
            validation_extra_wall_ms += float(
                full_reference_row.get("duration_ms", 0.0) or 0.0
            )
            validation_extra_backend_ms += execution_profile_backend_reported_ms(
                full_reference_row.get("execution_profile", {})
            )
            validation_extra_backend_calls += execution_profile_backend_calls(
                full_reference_row.get("execution_profile", {})
            )
        reference_wall_ms += float(
            full_reference_row.get("duration_ms", 0.0) or 0.0
        )
        reference_backend_ms += execution_profile_backend_reported_ms(
            full_reference_row.get("execution_profile", {})
        )
        reference_backend_calls += execution_profile_backend_calls(
            full_reference_row.get("execution_profile", {})
        )

        emitted_rows = [audited.low_row, *audited.audit_rows.values()]
        case_adaptive_families = set().union(
            *(_observed_families(row) for row in emitted_rows)
        )
        case_adaptive_confirmed = set().union(
            *(_confirmed_families(row) for row in emitted_rows)
        )
        case_reference_families = _observed_families(full_reference_row)
        case_reference_confirmed = _confirmed_families(full_reference_row)
        adaptive_families.update(case_adaptive_families)
        adaptive_confirmed_families.update(case_adaptive_confirmed)
        reference_families.update(case_reference_families)
        reference_confirmed_families.update(case_reference_confirmed)
        non_reproduced = _has_non_reproduced_candidate(emitted_rows)
        safety_after = runtime.complete_case(
            non_reproduced_candidate_increment=int(non_reproduced)
        )
        exact_comparison = CounterfactualCandidateComparison.compare(
            audited.low_row,
            full_reference_row,
        )
        case_results.append(
            {
                "trace_index": index,
                "campaign_seed": campaign_seed,
                "case_id": case.case_id,
                "case_sha256": case_hash,
                "low_backends": low_backends,
                "omitted_backends": omitted_backends,
                "low_variants": [variant.name for variant in low_variants],
                "omitted_reference_variants": [
                    variant.name for variant in omitted_variants
                ],
                "safety_state_before": safety_before.state,
                "safety_state_after": safety_after.state,
                "safety_reasons_before": list(safety_before.reasons),
                "safety_reasons_after": list(safety_after.reasons),
                "safety_propensities_before": {
                    axis: str(propensity)
                    for axis, propensity in safety_before.audit_propensities
                },
                "safety_propensities_after": {
                    axis: str(propensity)
                    for axis, propensity in safety_after.audit_propensities
                },
                "audit_plans": {
                    axis: plan.to_dict() for axis, plan in sorted(plans.items())
                },
                "adaptive": adaptive_summary,
                "full_reference_reused_from_joint_audit": (
                    full_reference_reused
                ),
                "full_reference_status": full_reference_row.get("status", ""),
                "full_reference_accounting": {
                    "wall_ms": float(
                        full_reference_row.get("duration_ms", 0.0) or 0.0
                    ),
                    "backend_reported_ms": (
                        execution_profile_backend_reported_ms(
                            full_reference_row.get("execution_profile", {})
                        )
                    ),
                    "backend_calls": execution_profile_backend_calls(
                        full_reference_row.get("execution_profile", {})
                    ),
                },
                "adaptive_candidate_families": sorted(case_adaptive_families),
                "full_reference_candidate_families": sorted(
                    case_reference_families
                ),
                "low_vs_full_reference": exact_comparison.to_dict(),
            }
        )

    harness_wall_s = time.perf_counter() - wall_started
    adaptive_roots = _roots(adaptive_families)
    reference_roots = _roots(reference_families)
    adaptive_confirmed_roots = _roots(adaptive_confirmed_families)
    reference_confirmed_roots = _roots(reference_confirmed_families)
    return {
        "schema_version": "rlcmf-paired-replay-result-v1",
        "manifest_id": manifest.payload["manifest_id"],
        "manifest_sha256": manifest.sha256,
        "measurement_scope": (
            "frozen_case_paired_shadow_candidate_evidence_not_confirmed_real_bugs"
        ),
        "policy": policy.to_dict(),
        "trace": {
            "case_count": len(cases),
            "case_sha256s": [entry["case_sha256"] for entry in schedule],
            "trace_sha256": canonical_payload_sha256(
                [entry["case_sha256"] for entry in schedule]
            ),
        },
        "fairness": {
            "frozen_seed_schedule_executed_without_replacement": True,
            "case_counts_by_seed": dict(sorted(runtime.case_counts.items())),
            "hot_cold_zero_result_rows_retained": True,
            "favorable_early_stopping_used": False,
        },
        "metrics": {
            "harness_wall_s_including_validation": harness_wall_s,
            "adaptive_accounted_wall_ms": adaptive_wall_ms,
            "adaptive_backend_reported_ms": adaptive_backend_ms,
            "adaptive_backend_calls": adaptive_backend_calls,
            "full_reference_wall_ms": reference_wall_ms,
            "full_reference_backend_reported_ms": reference_backend_ms,
            "full_reference_backend_calls": reference_backend_calls,
            "validation_extra_wall_ms_excluded_from_adaptive": (
                validation_extra_wall_ms
            ),
            "validation_extra_backend_reported_ms_excluded_from_adaptive": (
                validation_extra_backend_ms
            ),
            "validation_extra_backend_calls_excluded_from_adaptive": (
                validation_extra_backend_calls
            ),
            "adaptive_cases_per_backend_cpu_hour_proxy": (
                len(cases) / (adaptive_backend_ms / 3_600_000.0)
                if adaptive_backend_ms > 0
                else None
            ),
            "full_reference_cases_per_backend_cpu_hour_proxy": (
                len(cases) / (reference_backend_ms / 3_600_000.0)
                if reference_backend_ms > 0
                else None
            ),
            "candidate_family_recall": _recall(
                adaptive_families,
                reference_families,
            ),
            "candidate_root_recall": _recall(adaptive_roots, reference_roots),
            "confirmed_candidate_family_recall": _recall(
                adaptive_confirmed_families,
                reference_confirmed_families,
            ),
            "confirmed_candidate_root_recall": _recall(
                adaptive_confirmed_roots,
                reference_confirmed_roots,
            ),
            "adaptive_candidate_families": sorted(adaptive_families),
            "full_reference_candidate_families": sorted(reference_families),
            "adaptive_candidate_roots": sorted(adaptive_roots),
            "full_reference_candidate_roots": sorted(reference_roots),
            "independently_confirmed_unique_real_roots_per_cpu_hour": None,
        },
        "safety": {
            "final_state": runtime.current_safety_decision.state,
            "controller_log_integrity": runtime.controller.verify_log(),
            "pending_sentinel_plans": runtime.coordinator.to_dict()[
                "pending_plan_ids"
            ],
            "axis_evidence": {
                axis: tracker.snapshot().to_dict()
                for axis, tracker in sorted(
                    runtime.coordinator.safety_trackers.items()
                )
            },
        },
        "promotion": {
            "eligible": False,
            "reason": (
                "paired replay measures candidate evidence only; native reproduction "
                "and independent real-root triage remain required"
            ),
        },
        "case_results": case_results,
    }
