from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from datadiff.artifact import save_bug_artifact
from datadiff.backends import make_backend
from datadiff.backends.base import Backend
from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.feedback import FeedbackState
from datadiff.guidance import GuidanceState
from datadiff.metamorphic import build_metamorphic_variants, evaluate_metamorphic_variants
from datadiff.normalizer import normalize_result
from datadiff.oracle import PROBE_ROOTS, evaluate_case
from datadiff.operation_combo import classify_operation_combo
from datadiff.preflight import preflight_case
from datadiff.quality_oracles import evaluate_quality_oracles
from datadiff.reward import candidate_bug_family_keys, candidate_bug_signatures, row_reward_signals
from datadiff.scheduler import LocalSourceScheduler
from datadiff.targets import common_capabilities, describe_targets
from datadiff.util import CORPUS_DIR, RUNS_DIR, JsonlWriter, append_jsonl, dump_json, ensure_dirs, run_meta_path, utc_now

ProgressCallback = Callable[[dict[str, Any]], None]


def _case_summary(case_data: dict[str, Any]) -> dict[str, Any]:
    program = case_data.get("program", {})
    return {
        "case_id": case_data.get("case_id", ""),
        "seed": case_data.get("seed", 0),
        "table_count": len(case_data.get("tables", [])),
        "row_count": sum(len(table.get("rows", [])) for table in case_data.get("tables", [])),
        "program": {
            "program_id": program.get("program_id", ""),
            "seed": program.get("seed", 0),
            "operations": program.get("operations", []),
        },
    }


def _normalized_summary(normalized: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        backend: {
            "backend": data.get("backend", backend),
            "status": data.get("status", "unknown"),
            "columns": data.get("columns", []),
            "row_count": len(data.get("rows", [])),
            "error_type": data.get("error_type", ""),
            "error": data.get("error", ""),
        }
        for backend, data in normalized.items()
    }


def _raw_results_summary(raw_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        backend: {
            "backend": data.get("backend", backend),
            "status": data.get("status", "unknown"),
            "error_type": data.get("error_type", ""),
            "duration_ms": data.get("duration_ms", 0.0),
        }
        for backend, data in raw_results.items()
    }


def _guidance_summary(guidance: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": guidance.get("score", 0.0),
        "matched_targets": guidance.get("matched_targets", []),
        "candidate_count": guidance.get("candidate_count", 1),
        "contributing_candidate_count": guidance.get("contributing_candidate_count", guidance.get("candidate_count", 1)),
        "pruned_candidate_count": guidance.get("pruned_candidate_count", 0),
        "feature_count": len(guidance.get("features", [])),
        "frontier_bucket_count": len(guidance.get("frontier_buckets", [])),
        "path_coverage_proxy": guidance.get("score_breakdown", {}).get("path_coverage_proxy", 0.0),
        "data_sensitivity": guidance.get("score_breakdown", {}).get("data_sensitivity", 0.0),
        "frontier_conformance": guidance.get("score_breakdown", {}).get("frontier_conformance", 0.0),
        "contribution_potential": guidance.get("score_breakdown", {}).get("contribution_potential", 0.0),
        "combo_priority": guidance.get("score_breakdown", {}).get("combo_priority", 0.0),
        "online_weight_mean": guidance.get("score_breakdown", {}).get("online_weight_mean", 1.0),
        "online_weight_max": guidance.get("score_breakdown", {}).get("online_weight_max", 1.0),
        "online_weight_updates": guidance.get("score_breakdown", {}).get("online_weight_updates", 0.0),
        "family_saturation_penalty": guidance.get("score_breakdown", {}).get("family_saturation_penalty", 0.0),
        "family_saturation_active": guidance.get("score_breakdown", {}).get("family_saturation_active", 0.0),
        "issue_replay_saturation_penalty": guidance.get("score_breakdown", {}).get(
            "issue_replay_saturation_penalty", 0.0
        ),
        "issue_replay_saturation_active": guidance.get("score_breakdown", {}).get(
            "issue_replay_saturation_active", 0.0
        ),
        "issue_replay_global_saturation_penalty": guidance.get("score_breakdown", {}).get(
            "issue_replay_global_saturation_penalty", 0.0
        ),
        "issue_replay_global_saturation_active": guidance.get("score_breakdown", {}).get(
            "issue_replay_global_saturation_active", 0.0
        ),
    }


def _quality_oracle_summary(oracles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": oracle.get("name", "unknown"),
            "verdict": oracle.get("verdict", "unknown"),
            "passed": bool(oracle.get("passed", False)),
            "score": oracle.get("score", 0.0),
        }
        for oracle in oracles
    ]


def _generated_candidate_metadata(case: Case) -> dict[str, Any]:
    return {
        "seed_lineage": {
            "root_seed": case.seed,
            "parent_seed": None,
            "parent_case_id": "",
            "mutation_seed": None,
            "depth": 0,
        },
        "mutation": {
            "operator": "generated",
            "detail": "generated",
            "changed": False,
        },
    }


def _source_scheduler_snapshot(feedback: FeedbackState | None) -> list[dict[str, Any]]:
    scheduler = getattr(feedback, "source_scheduler", None) if feedback is not None else None
    if scheduler is None or not hasattr(scheduler, "snapshot"):
        return []
    return scheduler.snapshot()


def _compact_log_row(row: dict[str, Any], log_level: str) -> dict[str, Any]:
    if log_level == "full":
        return row
    has_findings = bool(row.get("findings"))
    if has_findings and log_level in {"compact", "minimal"}:
        # Finding rows keep reproduction detail; run-level duplicated metadata
        # stays in meta.json and artifacts.
        return {
            key: value
            for key, value in row.items()
            if key not in {"environment", "targets"}
        }

    out = {
        "run_at": row.get("run_at", ""),
        "status": row.get("status", "unknown"),
        "case": _case_summary(row.get("case", {})),
        "behavior_signature": row.get("behavior_signature", ""),
        "duration_ms": row.get("duration_ms", 0.0),
        "findings": row.get("findings", []),
        "bug_dir": row.get("bug_dir", ""),
        "candidate_source": row.get("candidate_source", "generated"),
        "seed_lineage": row.get("seed_lineage", {}),
        "mutation": row.get("mutation", {}),
        "operation_combo": row.get("operation_combo", {}),
        "source_reward": row.get("source_reward"),
        "source_scheduler": row.get("source_scheduler", []),
        "preflight": row.get("preflight", {}),
        "quality_oracles": _quality_oracle_summary(row.get("quality_oracles", [])),
        "guidance": _guidance_summary(row.get("guidance", {})),
        "candidate_seed_start": row.get("candidate_seed_start", 0),
        "candidate_pool_size": row.get("candidate_pool_size", 1),
        "case_index": row.get("case_index", 0),
        "elapsed_s": row.get("elapsed_s", 0.0),
        "is_new_behavior": row.get("is_new_behavior", False),
        "stored_in_feedback_corpus": row.get("stored_in_feedback_corpus", False),
        "feedback_corpus_persisted": row.get("feedback_corpus_persisted", False),
        "feedback_eligible": row.get("feedback_eligible", True),
        "feedback_skip_reason": row.get("feedback_skip_reason", ""),
        "feedback_record_skip_reason": row.get("feedback_record_skip_reason", ""),
    }
    if log_level == "compact":
        out["normalized"] = _normalized_summary(row.get("normalized", {}))
        out["raw_results"] = _raw_results_summary(row.get("raw_results", {}))
    else:
        out["backend_status"] = {
            backend: data.get("status", "unknown")
            for backend, data in row.get("normalized", {}).items()
        }
    return out


def behavior_signature(row: dict[str, Any]) -> str:
    payload = {
        "case_ops": row["case"]["program"]["operations"],
        "backend_status": {
            b: r["status"] for b, r in sorted(row["normalized"].items())
        },
        "normalized_shape": {
            b: {
                "columns": r.get("columns", []),
                "rows": len(r.get("rows", [])),
                "sample": r.get("rows", [])[:3],
                "error_type": r.get("error_type", ""),
            }
            for b, r in sorted(row["normalized"].items())
        },
        "finding_kinds": sorted(f["kind"] for f in row.get("findings", [])),
        "finding_roots": sorted(f.get("root_cause", "unknown") for f in row.get("findings", [])),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


CALIBRATION_PROBE_OPS = frozenset(PROBE_ROOTS) | {
    "running_sum",
    "tuple_absence_filter",
}


def _feedback_storage_decision(
    case: Case,
    *,
    candidate_source: str = "generated",
    seed_lineage: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    lineage_depth = 0
    if isinstance(seed_lineage, dict):
        lineage_depth = int(seed_lineage.get("depth", 0) or 0)
    if candidate_source == "feedback_mutation" or lineage_depth > 0:
        return False, "feedback_mutation_child"
    if any(str(op.get("op", "")) in CALIBRATION_PROBE_OPS for op in case.program.operations):
        return False, "calibration_probe_case"
    return True, ""


def _execute_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    backend_instances: dict[str, Backend] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_results = {}
    normalized = {}
    for backend_name in backends:
        backend = backend_instances[backend_name] if backend_instances is not None else make_backend(backend_name)
        result = backend.run(case.tables, case.program)
        raw_results[backend_name] = {
            k: v
            for k, v in result.to_dict().items()
            if k != "data"
        }
        normalized[backend_name] = normalize_result(
            result,
            case.program,
            enable_normalizer=config.enable_normalizer,
        )
    return raw_results, normalized


def run_loaded_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig | None = None,
    save_artifact: bool = True,
    backend_instances: dict[str, Backend] | None = None,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    started = time.perf_counter()
    raw_results, normalized = _execute_case(case, backends, config, backend_instances=backend_instances)

    findings = []
    if config.enable_differential_oracle:
        findings = evaluate_case(case, normalized)
    metamorphic_rows: dict[str, Any] = {}
    if config.enable_metamorphic_oracle:
        variant_results = {}
        for variant in build_metamorphic_variants(case, limit=max(0, config.metamorphic_variant_limit)):
            variant_raw, variant_norm = _execute_case(
                variant.case,
                backends,
                config,
                backend_instances=backend_instances,
            )
            variant_results[variant.name] = variant_norm
            metamorphic_rows[variant.name] = {
                "relation": variant.relation,
                "case": variant.case.to_dict(),
                "raw_results": variant_raw,
                "normalized": {k: v.to_dict() for k, v in variant_norm.items()},
            }
        findings.extend(evaluate_metamorphic_variants(case, normalized, variant_results))
    if findings:
        annotate_findings(
            case,
            findings,
            normalized=normalized,
            raw_results=raw_results,
            config=config.to_dict(),
            backends=backends,
        )

    row = {
        "run_at": utc_now(),
        "case": case.to_dict(),
        "targets": target_specs if target_specs is not None else describe_targets(backends),
        "raw_results": raw_results,
        "normalized": {k: v.to_dict() for k, v in normalized.items()},
        "metamorphic": metamorphic_rows,
        "findings": [f.to_dict() for f in findings],
        "config": config.to_dict(),
        "environment": environment if environment is not None else collect_environment(),
        "status": "bug" if findings else "ok",
        "duration_ms": (time.perf_counter() - started) * 1000,
    }
    row["behavior_signature"] = behavior_signature(row)
    if findings and save_artifact and config.enable_artifact:
        bug_dir = save_bug_artifact(
            case,
            raw_results=raw_results,
            normalized={k: v.to_dict() for k, v in normalized.items()},
            findings=findings,
            config=config.to_dict(),
        )
        row["bug_dir"] = str(bug_dir)
    return row


def run_fuzz(
    cases: int | None,
    seed: int,
    backends: list[str],
    config: ExperimentConfig | None = None,
    duration_s: float | None = None,
    save_cases: bool = False,
    case_log_file: Path | None = None,
    checkpoint_interval_s: float | None = None,
    progress_interval_s: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    ensure_dirs()
    config = config or ExperimentConfig()
    if cases is None and duration_s is None:
        cases = 100
    run_id = f"run-{utc_now().replace(':', '').replace('-', '').replace('Z', '')}-{time.time_ns()}"
    run_suffix = ".jsonl.gz" if config.compress_run_log else ".jsonl"
    run_file = RUNS_DIR / f"{run_id}{run_suffix}"
    if case_log_file is not None:
        save_cases = True
    resolved_case_log_file = case_log_file
    if save_cases and resolved_case_log_file is None:
        resolved_case_log_file = CORPUS_DIR / "generated" / f"{run_id}.cases.jsonl"
    checkpoint_file = RUNS_DIR / f"{run_id}.checkpoint.json" if checkpoint_interval_s is not None else None
    backend_instances = {backend_name: make_backend(backend_name) for backend_name in backends}
    environment = collect_environment()
    target_specs = describe_targets(backends)
    target_common_capabilities = common_capabilities(backends)
    seen: set[str] = set()
    feedback = (
        FeedbackState(
            persist_to_disk=config.persist_feedback_corpus,
            max_persisted=config.feedback_persist_limit,
            source_scheduler=(
                LocalSourceScheduler(
                    exploration_weight=config.local_source_exploration_weight,
                    enable_family_saturation=config.enable_family_saturation,
                    family_saturation_threshold=config.family_saturation_threshold,
                    saturated_family_reward=config.saturated_family_reward,
                    known_saturated_bug_families=config.known_saturated_bug_families,
                )
                if config.enable_local_source_scheduler
                else None
            ),
        )
        if config.enable_feedback
        else None
    )
    guided = config.guidance_strategy == "guided"
    candidate_pool = max(1, config.guidance_candidate_pool if guided else 1)
    guidance = (
        GuidanceState(
            targets=config.guidance_targets,
            enable_family_saturation=config.enable_family_saturation,
            family_saturation_threshold=config.family_saturation_threshold,
            family_saturation_penalty=config.family_saturation_penalty,
            saturated_family_reward=config.saturated_family_reward,
            known_saturated_bug_families=config.known_saturated_bug_families,
            issue_replay_saturation_threshold=config.issue_replay_saturation_threshold,
            issue_replay_saturation_penalty=config.issue_replay_saturation_penalty,
            issue_replay_global_saturation_threshold=config.issue_replay_global_saturation_threshold,
            issue_replay_global_saturation_penalty=config.issue_replay_global_saturation_penalty,
            active_backends=list(backends),
        )
        if guided
        else None
    )
    started = time.perf_counter()
    last_checkpoint = started
    last_progress = started
    executed = 0
    next_seed = seed
    findings_count = 0
    new_behavior_count = 0
    artifact_saved_count = 0
    preflight_repaired_count = 0
    preflight_fallback_count = 0
    preflight_invalid_count = 0
    quality_oracle_counts: dict[str, int] = {}

    def snapshot(status: str) -> dict[str, Any]:
        elapsed_s = time.perf_counter() - started
        return {
            "run_id": run_id,
            "status": status,
            "run_file": str(run_file),
            "case_log_file": str(resolved_case_log_file) if resolved_case_log_file is not None else "",
            "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else "",
            "requested_cases": cases,
            "executed_cases": executed,
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "throughput_cases_s": executed / elapsed_s if elapsed_s else 0.0,
            "findings": findings_count,
            "new_behavior_cases": new_behavior_count,
            "saved_artifacts": artifact_saved_count,
            "preflight": {
                "repaired_cases": preflight_repaired_count,
                "fallback_cases": preflight_fallback_count,
                "invalid_cases": preflight_invalid_count,
            },
            "quality_oracles": quality_oracle_counts,
            "seed": seed,
            "next_seed": next_seed,
            "guidance": {
                "strategy": config.guidance_strategy,
                "candidate_pool": candidate_pool,
                "targets": config.guidance_targets,
            },
            "backends": backends,
            "targets": target_specs,
            "common_capabilities": target_common_capabilities,
            "config": config.to_dict(),
            "environment": environment,
            "log_level": config.log_level,
            "updated_at": utc_now(),
        }

    def write_checkpoint(status: str) -> None:
        if checkpoint_file is not None:
            dump_json(snapshot(status), checkpoint_file)

    run_writer_context = JsonlWriter(run_file, compresslevel=1)
    run_writer = run_writer_context.__enter__()
    case_writer_context = JsonlWriter(resolved_case_log_file, compresslevel=1) if resolved_case_log_file else None
    case_writer = case_writer_context.__enter__() if case_writer_context is not None else None

    while True:
        if cases is not None and executed >= cases:
            break
        if duration_s is not None and executed > 0 and (time.perf_counter() - started) >= duration_s:
            break
        candidate_seed_start = next_seed
        candidates: list[Case] = []
        candidate_meta: dict[int, dict[str, Any]] = {}
        for offset in range(candidate_pool):
            case_seed = candidate_seed_start + offset
            generated = generate_case(
                case_seed,
                type_aware=config.enable_type_aware_generation,
                profile=config.generator_profile,
            )
            selected = feedback.choose_case(case_seed, generated) if feedback is not None else generated
            source = getattr(feedback, "last_candidate_source", "generated") if feedback is not None else "generated"
            metadata = (
                getattr(feedback, "last_candidate_metadata", None)
                if feedback is not None
                else None
            ) or selected.metadata or _generated_candidate_metadata(generated)
            preflight = preflight_case(
                selected,
                enable_validation=config.enable_preflight_validation,
                enable_repair=config.enable_preflight_repair,
            )
            candidate = preflight.case
            candidate_meta[id(candidate)] = {
                "source": source,
                "generated_seed": case_seed,
                "seed_lineage": metadata.get("seed_lineage", {}),
                "mutation": metadata.get("mutation", {}),
                "operation_combo": classify_operation_combo(candidate.program.operations),
                "preflight": preflight.to_dict(),
            }
            candidates.append(candidate)
        next_seed += candidate_pool
        if guidance is not None:
            decision = guidance.choose_case(candidates)
            case = decision.case
            guidance_row = decision.to_dict()
        else:
            case = candidates[0]
            guidance_row = {
                "score": 0.0,
                "features": [],
                "matched_targets": [],
                "candidate_count": 1,
            }
        selected_meta = candidate_meta.get(
            id(case),
            {
                "source": "generated",
                "generated_seed": case.seed,
                "seed_lineage": _generated_candidate_metadata(case)["seed_lineage"],
                "mutation": _generated_candidate_metadata(case)["mutation"],
                "operation_combo": classify_operation_combo(case.program.operations),
                "preflight": {
                    "valid": True,
                    "repaired": False,
                    "fallback_used": False,
                    "errors_before": [],
                    "errors_after": [],
                },
            },
        )
        preflight_row = selected_meta["preflight"]
        case_seed = case.seed
        if case_writer is not None:
            case_writer.write(
                {
                    "run_id": run_id,
                    "case_index": executed,
                    "seed": case_seed,
                    "candidate_seed_start": candidate_seed_start,
                    "candidate_pool_size": candidate_pool,
                    "guidance": guidance_row,
                    "candidate_source": selected_meta["source"],
                    "seed_lineage": selected_meta["seed_lineage"],
                    "mutation": selected_meta["mutation"],
                    "operation_combo": selected_meta["operation_combo"],
                    "preflight": preflight_row,
                    "generated_at": utc_now(),
                    "case": case.to_dict(),
                }
            )
        save_artifact_for_case = _artifact_budget_available(config, artifact_saved_count)
        row = run_loaded_case(
            case,
            backends=backends,
            config=config,
            save_artifact=not config.enable_reducer and save_artifact_for_case,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
        )
        if row["findings"] and config.enable_reducer:
            from datadiff.reducer import reduce_case

            reduced = reduce_case(
                case,
                backends=backends,
                config=ExperimentConfig(
                    enable_type_aware_generation=config.enable_type_aware_generation,
                    enable_normalizer=config.enable_normalizer,
                    enable_differential_oracle=config.enable_differential_oracle,
                    enable_metamorphic_oracle=config.enable_metamorphic_oracle,
                    enable_feedback=False,
                    enable_reducer=False,
                    enable_artifact=False,
                    oracle_mode=config.oracle_mode,
                    generator_profile=config.generator_profile,
                    metamorphic_variant_limit=config.metamorphic_variant_limit,
                ),
                target_kinds=[finding["kind"] for finding in row["findings"]],
                target_roots=[finding.get("root_cause", "unknown") for finding in row["findings"]],
            )
            reduced_row = run_loaded_case(
                reduced,
                backends=backends,
                config=config,
                save_artifact=_artifact_budget_available(config, artifact_saved_count),
                backend_instances=backend_instances,
                environment=environment,
                target_specs=target_specs,
            )
            reduced_row["original_case"] = case.to_dict()
            reduced_row["reduction"] = {
                "original_rows": len(case.tables[0].rows),
                "reduced_rows": len(reduced.tables[0].rows),
                "original_ops": len(case.program.operations),
                "reduced_ops": len(reduced.program.operations),
            }
            row = reduced_row
        if row.get("findings") and row.get("bug_dir"):
            artifact_saved_count += 1
            row["artifact_saved"] = True
        elif row.get("findings") and config.enable_artifact:
            row["artifact_saved"] = False
            row["artifact_skipped_reason"] = "artifact_limit_reached"

        sig = row["behavior_signature"]
        row["is_new_behavior"] = sig not in seen
        seen.add(sig)
        if feedback is not None:
            reward_signals = row_reward_signals(row)
            row_candidate_families = list(candidate_bug_family_keys(row.get("findings") or []))
            row_candidate_signatures = list(candidate_bug_signatures(row.get("findings") or []))
            feedback_eligible, feedback_skip_reason = _feedback_storage_decision(
                case,
                candidate_source=selected_meta["source"],
                seed_lineage=selected_meta["seed_lineage"],
            )
            row["feedback_eligible"] = feedback_eligible
            row["feedback_skip_reason"] = feedback_skip_reason
            if feedback_eligible:
                row["stored_in_feedback_corpus"] = feedback.record(
                    case,
                    sig,
                    bool(row["findings"]),
                    candidate_bug_families=row_candidate_families,
                )
                row["feedback_corpus_persisted"] = feedback.last_persisted_to_disk
                row["feedback_record_skip_reason"] = feedback.last_record_skip_reason
            else:
                feedback.last_persisted_to_disk = False
                row["stored_in_feedback_corpus"] = False
                row["feedback_corpus_persisted"] = False
                row["feedback_record_skip_reason"] = ""
            row["source_reward"] = feedback.record_candidate_result(
                selected_meta["source"],
                has_finding=bool(row["findings"]),
                is_new_behavior=bool(row["is_new_behavior"]),
                preflight=preflight_row,
                candidate_bug=bool(reward_signals["candidate_bug"]),
                semantic_divergence=bool(reward_signals["semantic_divergence"]),
                false_positive=bool(reward_signals["false_positive"]),
                candidate_bug_families=row_candidate_families,
                candidate_bug_signatures=row_candidate_signatures,
            )
            row["source_scheduler"] = _source_scheduler_snapshot(feedback)
        else:
            row["stored_in_feedback_corpus"] = False
            row["feedback_corpus_persisted"] = False
            row["feedback_eligible"] = False
            row["feedback_skip_reason"] = "feedback_disabled"
            row["feedback_record_skip_reason"] = ""
            row["source_reward"] = None
            row["source_scheduler"] = []
        if guidance is not None:
            guidance.record_result(case, row)
        row["guidance"] = guidance_row
        row["candidate_source"] = selected_meta["source"]
        row["seed_lineage"] = selected_meta["seed_lineage"]
        row["mutation"] = selected_meta["mutation"]
        row["operation_combo"] = selected_meta["operation_combo"]
        row["preflight"] = preflight_row
        row["candidate_seed_start"] = candidate_seed_start
        row["candidate_pool_size"] = candidate_pool
        row["case_index"] = executed
        row["elapsed_s"] = round(time.perf_counter() - started, 6)
        quality_oracles = evaluate_quality_oracles(
            case,
            row,
            candidate_source=selected_meta["source"],
            preflight=preflight_row,
            guidance_decision=guidance_row,
            guidance_strategy=config.guidance_strategy,
            guidance_targets=config.guidance_targets,
        )
        row["quality_oracles"] = [oracle.to_dict() for oracle in quality_oracles]
        for oracle in row["quality_oracles"]:
            quality_oracle_counts[f"{oracle['name']}:{oracle['verdict']}"] = (
                quality_oracle_counts.get(f"{oracle['name']}:{oracle['verdict']}", 0) + 1
            )
        preflight_repaired_count += int(bool(preflight_row.get("repaired", False)))
        preflight_fallback_count += int(bool(preflight_row.get("fallback_used", False)))
        preflight_invalid_count += int(not bool(preflight_row.get("valid", True)))
        findings_count += len(row["findings"])
        new_behavior_count += int(row["is_new_behavior"])
        run_writer.write(_compact_log_row(row, config.log_level))
        executed += 1

        now = time.perf_counter()
        if checkpoint_interval_s is not None and (now - last_checkpoint) >= checkpoint_interval_s:
            write_checkpoint("running")
            last_checkpoint = now
        if (
            progress_callback is not None
            and progress_interval_s is not None
            and (now - last_progress) >= progress_interval_s
        ):
            progress_callback(snapshot("running"))
            last_progress = now

        if duration_s is not None and (time.perf_counter() - started) >= duration_s:
            break

    if case_writer_context is not None:
        case_writer_context.__exit__(None, None, None)
    run_writer_context.__exit__(None, None, None)

    elapsed_s = time.perf_counter() - started
    meta = snapshot("completed")
    meta["elapsed_s"] = elapsed_s
    meta["throughput_cases_s"] = executed / elapsed_s if elapsed_s else 0.0
    dump_json(meta, run_meta_path(run_file))
    write_checkpoint("completed")
    if progress_callback is not None:
        progress_callback(meta)
    return run_file


def _artifact_budget_available(config: ExperimentConfig, saved_count: int) -> bool:
    if not config.enable_artifact:
        return False
    if config.artifact_limit is None:
        return True
    return saved_count < max(0, int(config.artifact_limit))
