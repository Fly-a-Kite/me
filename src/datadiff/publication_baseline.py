from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PUBLICATION_BASELINE_SCHEMA_VERSION = "icse-fse-publication-baseline-index-v1"
STANDARD_EVIDENCE_ROLES = ("protocol", "result", "analysis", "report", "runtime")
DEFAULT_INDEX_PATH = Path("experiments/publication_baseline/v1/index.json")
DEFAULT_CHECKSUM_PATH = Path("experiments/publication_baseline/v1/checksums.sha256")
P2_FROZEN_SOURCE_ROOT = Path(
    "experiments/publication_baseline/v1/p2_frozen_source_v2"
)
P2_FROZEN_SOURCE_PATHS = (
    "archive_manifest.json",
    "scripts/run_ccs_obligation_priority_ablation.py",
    "src/datadiff/ccs_priority_ablation.py",
    "src/datadiff/contract_comparison.py",
    "src/datadiff/metamorphic.py",
    "src/datadiff/method_arms.py",
    "src/datadiff/obligation_priority.py",
    "src/datadiff/run_loaded.py",
    "src/datadiff/test_obligations.py",
)


GROUP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "group_id": "p0-p1-recomputed-verification-v1",
        "stage": "P0-P1",
        "title": "Semantic/measurement foundation and research-control verification",
        "evidence_status": "recomputed_verification_with_historical_log_limitation",
        "claim_boundary": (
            "Supports cost attribution and control-plane invariants. The 24-case replay "
            "is diagnostic and cannot support recall or bug-yield claims."
        ),
        "artifacts": {
            "protocol": ["experiments/p0_p1_archive_verification_v1_protocol.json"],
            "result": ["experiments/p0_p1_archive_verification_v1_result.json"],
            "analysis": ["experiments/p0_p1_archive_verification_v1_analysis.json"],
            "report": ["experiments/p0_p1_archive_verification_v1_report.md"],
            "runtime": ["experiments/p0_p1_archive_verification_v1_runtime.txt"],
            "source_report": [
                "docs/latticefuzz_p0_implementation_report.md",
                "docs/latticefuzz_p1_research_harness_report.md",
            ],
            "frozen_input": [
                "experiments/publication_baseline/v1/p0_diagnostic_replay_v1/frozen_cases.jsonl",
                "experiments/publication_baseline/v1/p0_diagnostic_replay_v1/source_run.jsonl.gz",
            ],
            "diagnostic_replay": [
                "experiments/publication_baseline/v1/p0_diagnostic_replay_v1/full6_result.json",
                "experiments/publication_baseline/v1/p0_diagnostic_replay_v1/sampled1_result.json",
            ],
        },
    },
    {
        "group_id": "p2-ccs-ir-paired-equivalence-v1",
        "stage": "P2.1",
        "title": "CCS-IR paired representation equivalence",
        "evidence_status": "completed_negative_performance_claim",
        "claim_boundary": (
            "Supports behavior preservation on the frozen paired corpus; does not "
            "support a speedup or real-bug-yield claim."
        ),
        "artifacts": {
            "protocol": ["experiments/ccs_ir_paired_ablation_100x5_v1_preregistration.json"],
            "result": ["experiments/ccs_ir_paired_ablation_100x5_v1_result.json"],
            "analysis": ["experiments/ccs_ir_paired_ablation_100x5_v1_analysis.json"],
            "report": ["experiments/ccs_ir_paired_ablation_100x5_v1_report.md"],
            "runtime": ["experiments/ccs_ir_paired_ablation_100x5_v1_runtime.txt"],
            "frozen_input": ["experiments/ccs_ir_frozen_corpus_seeds1001_1100_v1.json"],
            "diagnostic": ["experiments/ccs_ir_paired_ablation_100x5_v1_case1010_diagnostic.json"],
        },
    },
    {
        "group_id": "p2-ccs-ir-fresh-process-v1",
        "stage": "P2.1",
        "title": "CCS-IR fresh-process replication",
        "evidence_status": "completed_inconclusive_performance",
        "claim_boundary": (
            "Supports behavior preservation across fresh processes; observed overhead "
            "direction is descriptive because confidence intervals include one."
        ),
        "artifacts": {
            "protocol": ["experiments/ccs_ir_fresh_process_10x100x5_v1_preregistration.json"],
            "result": ["experiments/ccs_ir_fresh_process_10x100x5_v1_result.json"],
            "analysis": ["experiments/ccs_ir_fresh_process_10x100x5_v1_analysis.json"],
            "report": ["experiments/ccs_ir_fresh_process_10x100x5_v1_report.md"],
            "runtime": ["experiments/ccs_ir_fresh_process_10x100x5_v1_runtime.txt"],
        },
        "globs": [
            {
                "role": "worker_result",
                "pattern": "experiments/ccs_ir_fresh_process_10x100x5_v1_workers/*.json",
                "expected_count": 20,
            }
        ],
    },
    {
        "group_id": "p2-obligation-development-coverage-v2",
        "stage": "P2.2-P2.4",
        "title": "CCS obligation development-corpus coverage",
        "evidence_status": "completed_development_corpus",
        "claim_boundary": (
            "Supports representation completeness on the development corpus only; "
            "does not establish unseen-program generalization or bug yield."
        ),
        "artifacts": {
            "protocol": ["experiments/ccs_obligation_coverage_100_v2_protocol.json"],
            "result": ["experiments/ccs_obligation_coverage_100_v2.json"],
            "analysis": ["experiments/ccs_obligation_coverage_100_v2_analysis.json"],
            "report": ["experiments/ccs_obligation_coverage_100_v2_report.md"],
            "runtime": ["experiments/ccs_obligation_coverage_100_v2_runtime.txt"],
            "raw_report": ["experiments/ccs_obligation_coverage_100_v2.md"],
            "frozen_input": ["experiments/ccs_ir_frozen_corpus_seeds1001_1100_v1.json"],
        },
    },
    {
        "group_id": "p2-obligation-independent-holdout-v2",
        "stage": "P2.5",
        "title": "CCS obligation independent 500-case holdout coverage",
        "evidence_status": "completed_all_coverage_gates_passed",
        "claim_boundary": (
            "Supports representation completeness and exact default order on the "
            "independent holdout; does not support priority or bug-yield improvement."
        ),
        "artifacts": {
            "protocol": ["experiments/ccs_obligation_holdout_500_v2_protocol.json"],
            "result": ["experiments/ccs_obligation_holdout_500_v2_coverage.json"],
            "analysis": ["experiments/ccs_obligation_holdout_500_v2_analysis.json"],
            "report": ["experiments/ccs_obligation_holdout_500_v2_coverage.md"],
            "runtime": ["experiments/ccs_obligation_holdout_500_v2_coverage_runtime.txt"],
            "manifest": ["experiments/ccs_obligation_holdout_500_v2_manifest.json"],
            "freeze_runtime": ["experiments/ccs_obligation_holdout_500_v2_freeze_runtime.txt"],
            "frozen_input": ["experiments/ccs_obligation_holdout_seeds940001_940500_v2.json"],
            "implementation_manifest": ["experiments/ccs_obligation_priority_v2_implementation.json"],
            "frozen_source": [
                (P2_FROZEN_SOURCE_ROOT / path).as_posix()
                for path in P2_FROZEN_SOURCE_PATHS
            ],
        },
    },
    {
        "group_id": "p2-obligation-priority-holdout-v2",
        "stage": "P2.6-P2.8",
        "title": "CCS obligation-priority equal-work holdout experiment",
        "evidence_status": "completed_zero_confirmed_root_delta",
        "claim_boundary": (
            "Both arms found the same independently triaged DataFusion root. The "
            "confirmed-root delta is zero, so risk priority is retained as a negative control."
        ),
        "artifacts": {
            "protocol": ["experiments/ccs_obligation_priority_holdout_500x5_v2_preregistration.json"],
            "result": ["experiments/ccs_obligation_priority_holdout_500x5_v2_result.json"],
            "analysis": ["experiments/ccs_obligation_priority_holdout_500x5_v2_analysis.json"],
            "report": ["experiments/ccs_obligation_priority_holdout_500x5_v2_report.md"],
            "runtime": ["experiments/ccs_obligation_priority_holdout_500x5_v2_runtime.txt"],
            "confirmation": [
                "experiments/ccs_obligation_priority_holdout_500x5_v2_confirmation_protocol.json",
                "experiments/ccs_obligation_priority_holdout_500x5_v2_confirmation_result.json",
                "experiments/ccs_obligation_priority_holdout_500x5_v2_confirmation_runtime.txt",
            ],
            "native_reproduction": [
                "experiments/ccs_obligation_priority_holdout_500x5_v2_native_reproduction.json",
                "experiments/ccs_obligation_priority_holdout_500x5_v2_native_reproduction_runtime.txt",
            ],
            "duplicate_search": ["experiments/ccs_obligation_priority_holdout_500x5_v2_external_issue_search.json"],
            "root_record": [
                "new_issue/datafusion_order_by_offset_subquery_groupby.md",
                "new_issue/datafusion_order_by_offset_subquery_groupby_body.md",
                "new_issue/datafusion_order_by_offset_subquery_groupby_reproducer.py",
            ],
        },
        "globs": [
            {
                "role": "worker_result",
                "pattern": "experiments/ccs_obligation_priority_holdout_500x5_v2_workers/*.json",
                "expected_count": 10,
            }
        ],
    },
    {
        "group_id": "rlcmf-large96-negative-replay-v1",
        "stage": "P3.4/P7",
        "title": "RLCMF 96-case paired negative replay",
        "evidence_status": "completed_regressed_diagnostic_only",
        "claim_boundary": (
            "Supports a negative cost result only. Zero candidate/root observations "
            "cannot establish recall or real-bug yield, and original process CPU was not recorded."
        ),
        "artifacts": {
            "protocol": ["experiments/publication_baseline/v1/rlcmf_large96_v1/manifest.json"],
            "result": ["experiments/publication_baseline/v1/rlcmf_large96_v1/result.json"],
            "analysis": [
                "experiments/publication_baseline/v1/rlcmf_large96_v1/analysis.json",
                "experiments/publication_baseline/v1/rlcmf_large96_v1/analysis_recomputed.json",
            ],
            "report": [
                "experiments/publication_baseline/v1/rlcmf_large96_v1/report.md",
                "experiments/publication_baseline/v1/rlcmf_large96_v1/report_recomputed.md",
            ],
            "runtime": ["experiments/publication_baseline/v1/rlcmf_large96_v1/runtime.txt"],
            "selection": ["experiments/publication_baseline/v1/rlcmf_large96_v1/selection.json"],
            "frozen_input": ["experiments/publication_baseline/v1/rlcmf_large96_v1/trace.jsonl"],
        },
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_publication_baseline_index(
    repo_root: Path,
    *,
    frozen_at: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    uses_by_path: dict[str, set[tuple[str, str]]] = defaultdict(set)
    groups: list[dict[str, Any]] = []
    key_facts = _load_key_facts(repo_root)

    for spec in GROUP_SPECS:
        group_id = str(spec["group_id"])
        artifacts = {
            str(role): [str(path) for path in paths]
            for role, paths in spec["artifacts"].items()
        }
        for glob_spec in spec.get("globs", []):
            role = str(glob_spec["role"])
            matches = sorted(
                path.relative_to(repo_root).as_posix()
                for path in repo_root.glob(str(glob_spec["pattern"]))
                if path.is_file()
            )
            expected_count = int(glob_spec["expected_count"])
            if len(matches) != expected_count:
                raise ValueError(
                    f"{group_id}:{role} expected {expected_count} files, found {len(matches)}"
                )
            artifacts.setdefault(role, []).extend(matches)

        for role, paths in artifacts.items():
            for relative_path in paths:
                _require_repository_relative_path(relative_path)
                path = repo_root / relative_path
                if not path.is_file():
                    raise FileNotFoundError(relative_path)
                uses_by_path[relative_path].add((group_id, role))

        groups.append(
            {
                "group_id": group_id,
                "stage": spec["stage"],
                "title": spec["title"],
                "evidence_status": spec["evidence_status"],
                "claim_boundary": spec["claim_boundary"],
                "standard_roles_complete": all(
                    bool(artifacts.get(role)) for role in STANDARD_EVIDENCE_ROLES
                ),
                "artifacts": artifacts,
                "key_facts": key_facts[group_id],
            }
        )

    files = []
    for relative_path in sorted(uses_by_path):
        path = repo_root / relative_path
        files.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "uses": [
                    {"group_id": group_id, "role": role}
                    for group_id, role in sorted(uses_by_path[relative_path])
                ],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": PUBLICATION_BASELINE_SCHEMA_VERSION,
        "baseline_id": "icse-fse-p3-publication-baseline-v1",
        "frozen_at": frozen_at,
        "scope": {
            "stages": ["P0", "P1", "P2", "RLCMF negative evidence"],
            "repository_local_only": True,
            "primary_purpose": (
                "Prevent later optimization from rewriting or losing historical baseline evidence."
            ),
        },
        "archival_policy": {
            "standard_roles": list(STANDARD_EVIDENCE_ROLES),
            "historical_negative_results_retained": True,
            "missing_measurements_are_explicit": True,
            "critical_evidence_only_in_tmp_allowed": False,
            "claim_upgrade_from_candidates_allowed": False,
            "measured_source_copied_before_p4_live_edits": True,
        },
        "groups": groups,
        "files": files,
        "summary": {
            "group_count": len(groups),
            "file_count": len(files),
            "total_bytes": sum(int(item["size_bytes"]) for item in files),
            "all_standard_roles_complete": all(
                bool(group["standard_roles_complete"]) for group in groups
            ),
            "repository_local_file_count": len(files),
            "critical_tmp_path_count": 0,
            "rlcmf_original_process_cpu_available": False,
            "p3_4_exit_ready": True,
        },
        "known_limitations": [
            "Original P0/P1 pytest terminal logs were not preserved; the indexed verification is explicitly recomputed.",
            "The 24-case P0/P1 replay is diagnostic and cannot support recall or bug-yield claims.",
            "The RLCMF six-file archive omitted the original shell timing log and human-readable environment payload; process CPU and package versions remain unavailable.",
            "RLCMF observed zero candidate families and roots, so no candidate/root recall or confirmed-root-yield claim is available.",
        ],
    }
    payload["index_digest"] = canonical_json_sha256(payload)
    return payload


def write_publication_baseline(
    repo_root: Path,
    index: Mapping[str, Any],
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    checksum_path: Path = DEFAULT_CHECKSUM_PATH,
) -> None:
    repo_root = repo_root.resolve()
    index_target = _resolve(repo_root, index_path)
    checksum_target = _resolve(repo_root, checksum_path)
    index_target.parent.mkdir(parents=True, exist_ok=True)
    index_target.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_entries = {
        str(item["path"]): str(item["sha256"])
        for item in index.get("files", [])
        if isinstance(item, Mapping)
    }
    index_relative = index_target.relative_to(repo_root).as_posix()
    checksum_entries[index_relative] = sha256_file(index_target)
    checksum_target.write_text(
        "".join(
            f"{digest}  {path}\n"
            for path, digest in sorted(checksum_entries.items())
        ),
        encoding="utf-8",
    )


def validate_publication_baseline(
    repo_root: Path,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    checksum_path: Path = DEFAULT_CHECKSUM_PATH,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    index_target = _resolve(repo_root, index_path)
    checksum_target = _resolve(repo_root, checksum_path)
    if not index_target.is_file():
        return {"valid": False, "errors": [f"missing index: {index_target}"], "warnings": [], "summary": {}}

    try:
        index = json.loads(index_target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"invalid index: {exc}"], "warnings": [], "summary": {}}

    if index.get("schema_version") != PUBLICATION_BASELINE_SCHEMA_VERSION:
        errors.append("unexpected schema_version")
    stored_digest = str(index.get("index_digest", ""))
    digest_payload = dict(index)
    digest_payload.pop("index_digest", None)
    observed_digest = canonical_json_sha256(digest_payload)
    if stored_digest != observed_digest:
        errors.append("index_digest mismatch")

    indexed_files: dict[str, dict[str, Any]] = {}
    for row in index.get("files", []):
        if not isinstance(row, Mapping):
            errors.append("non-object file row")
            continue
        relative_path = str(row.get("path", ""))
        try:
            _require_repository_relative_path(relative_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if relative_path in indexed_files:
            errors.append(f"duplicate indexed path: {relative_path}")
            continue
        path = repo_root / relative_path
        if not path.is_file():
            errors.append(f"missing indexed file: {relative_path}")
            continue
        observed_sha256 = sha256_file(path)
        if observed_sha256 != row.get("sha256"):
            errors.append(f"checksum mismatch: {relative_path}")
        if path.stat().st_size != row.get("size_bytes"):
            errors.append(f"size mismatch: {relative_path}")
        indexed_files[relative_path] = dict(row)

    groups = index.get("groups", [])
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            errors.append("non-object group row")
            continue
        group_id = str(group.get("group_id", ""))
        if not group_id or group_id in group_ids:
            errors.append(f"invalid or duplicate group_id: {group_id}")
        group_ids.add(group_id)
        artifacts = group.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            errors.append(f"{group_id}: artifacts must be an object")
            continue
        for role in STANDARD_EVIDENCE_ROLES:
            if not artifacts.get(role):
                errors.append(f"{group_id}: missing standard role {role}")
        for role, paths in artifacts.items():
            if not isinstance(paths, list):
                errors.append(f"{group_id}:{role} must be a list")
                continue
            for relative_path in paths:
                if str(relative_path) not in indexed_files:
                    errors.append(f"{group_id}:{role} path is not indexed: {relative_path}")

    _validate_checksum_manifest(
        repo_root,
        index_target,
        checksum_target,
        indexed_files,
        errors,
    )
    _validate_p0_archived_inputs(repo_root, errors)
    _validate_p2_frozen_source_archive(repo_root, errors)
    _validate_rlcmf_reanalysis(repo_root, errors)

    runtime_text = (
        repo_root / "experiments/publication_baseline/v1/rlcmf_large96_v1/runtime.txt"
    ).read_text(encoding="utf-8")
    if "original_process_cpu_seconds=not_recorded" not in runtime_text:
        errors.append("RLCMF missing process-CPU limitation is not explicit")
    if "human_readable_environment_versions=not_preserved" not in runtime_text:
        errors.append("RLCMF environment limitation is not explicit")

    summary = {
        "group_count": len(groups),
        "file_count": len(indexed_files),
        "checksum_entry_count": _checksum_entry_count(checksum_target),
        "all_standard_roles_complete": not any(
            "missing standard role" in error for error in errors
        ),
        "critical_evidence_only_in_tmp": False,
        "rlcmf_reanalysis_equivalent": not any(
            error.startswith("RLCMF") for error in errors
        ),
        "p3_4_exit_ready": not errors,
    }
    if index.get("summary", {}).get("rlcmf_original_process_cpu_available") is not False:
        warnings.append("index does not preserve the RLCMF process-CPU absence")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def _load_key_facts(repo_root: Path) -> dict[str, dict[str, Any]]:
    p0 = _load_json(repo_root / "experiments/p0_p1_archive_verification_v1_result.json")
    paired_analysis = _load_json(repo_root / "experiments/ccs_ir_paired_ablation_100x5_v1_analysis.json")
    fresh = _load_json(repo_root / "experiments/ccs_ir_fresh_process_10x100x5_v1_result.json")
    fresh_analysis = _load_json(repo_root / "experiments/ccs_ir_fresh_process_10x100x5_v1_analysis.json")
    development = _load_json(repo_root / "experiments/ccs_obligation_coverage_100_v2.json")
    development_analysis = _load_json(repo_root / "experiments/ccs_obligation_coverage_100_v2_analysis.json")
    holdout = _load_json(repo_root / "experiments/ccs_obligation_holdout_500_v2_coverage.json")
    holdout_analysis = _load_json(repo_root / "experiments/ccs_obligation_holdout_500_v2_analysis.json")
    priority = _load_json(repo_root / "experiments/ccs_obligation_priority_holdout_500x5_v2_result.json")
    priority_analysis = _load_json(repo_root / "experiments/ccs_obligation_priority_holdout_500x5_v2_analysis.json")
    rlcmf = _load_json(repo_root / "experiments/publication_baseline/v1/rlcmf_large96_v1/analysis.json")
    rlcmf_result = _load_json(repo_root / "experiments/publication_baseline/v1/rlcmf_large96_v1/result.json")

    control, treatment = priority["arm_ids"]
    return {
        "p0-p1-recomputed-verification-v1": {
            "targeted_tests_passed": p0["targeted_tests"]["passed"],
            "full6_backend_calls": p0["replay"]["full6"]["backend_calls"],
            "sampled1_backend_calls": p0["replay"]["sampled1"]["backend_calls"],
            "full6_backend_stage_share": p0["replay"]["full6"]["backend_stage_share"],
            "sampled1_backend_stage_share": p0["replay"]["sampled1"]["backend_stage_share"],
            "recall_claim_allowed": False,
        },
        "p2-ccs-ir-paired-equivalence-v1": {
            "paired_cases": 100,
            "backends": 5,
            "all_validity_gates_passed": paired_analysis["validity_gates"]["all_preregistered_gates_passed"],
            "outcome_equivalence_rate": paired_analysis["validity_gates"]["outcome_equivalence_rate"],
            "backend_call_difference_pairs": paired_analysis["validity_gates"]["pairs_with_backend_call_difference"],
            "performance_speedup": paired_analysis["conclusion"]["performance_speedup"],
        },
        "p2-ccs-ir-fresh-process-v1": {
            "case_pair_count": fresh["case_pair_count"],
            "worker_count": len(fresh["worker_artifacts"]),
            "all_validity_gates_passed": fresh_analysis["validity_gates"]["all_preregistered_gates_passed"],
            "mean_process_cpu_ratio": fresh["summary"]["mean_process_cpu_ratio"],
            "performance_significance": fresh_analysis["conclusion"]["performance_significance"],
        },
        "p2-obligation-development-coverage-v2": {
            "case_count": development["case_count"],
            "variant_instance_recall": development["summary"]["variant_instance_recall"],
            "relation_opportunity_recall": development["summary"]["relation_opportunity_recall"],
            "exact_variant_order_match_rate": development["summary"]["exact_variant_order_match_rate"],
            "all_coverage_gates_passed": development_analysis["acceptance_gates"]["all_gates_passed"],
            "independent_holdout": False,
        },
        "p2-obligation-independent-holdout-v2": {
            "case_count": holdout["case_count"],
            "variant_instance_recall": holdout["summary"]["variant_instance_recall"],
            "relation_opportunity_recall": holdout["summary"]["relation_opportunity_recall"],
            "exact_variant_order_match_rate": holdout["summary"]["exact_variant_order_match_rate"],
            "registered_obligation_constructibility_rate": holdout["summary"]["registered_obligation_constructibility_rate"],
            "all_coverage_gates_passed": holdout_analysis["acceptance_gates"]["all_gates_passed"],
        },
        "p2-obligation-priority-holdout-v2": {
            "case_pair_count": priority["case_pair_count"],
            "control_backend_calls": priority["arms"][control]["backend_calls"],
            "treatment_backend_calls": priority["arms"][treatment]["backend_calls"],
            "control_selected_variants": priority["arms"][control]["selected_variant_count"],
            "treatment_selected_variants": priority["arms"][treatment]["selected_variant_count"],
            "control_confirmed_roots": priority_analysis["bug_yield_effect"]["control_confirmed_independent_roots"],
            "treatment_confirmed_roots": priority_analysis["bug_yield_effect"]["treatment_confirmed_independent_roots"],
            "confirmed_root_delta": priority_analysis["bug_yield_effect"]["confirmed_independent_root_delta"],
            "shared_roots": priority_analysis["bug_yield_effect"]["shared_roots"],
        },
        "rlcmf-large96-negative-replay-v1": {
            "case_count": rlcmf["trace"]["case_count"],
            "backend_calls_ratio": rlcmf["metrics"]["backend_calls"]["ratio_of_totals"],
            "backend_calls_ratio_ci": rlcmf["metrics"]["backend_calls"]["ratio_confidence_interval"],
            "backend_work_ratio": rlcmf["metrics"]["backend_reported_ms"]["ratio_of_totals"],
            "wall_ratio": rlcmf["metrics"]["wall_ms"]["ratio_of_totals"],
            "primary_decision": rlcmf["primary_decision"]["state"],
            "candidate_family_count": len(rlcmf_result["metrics"]["adaptive_candidate_families"]),
            "candidate_root_count": len(rlcmf_result["metrics"]["adaptive_candidate_roots"]),
            "original_process_cpu_available": False,
        },
    }


def _validate_checksum_manifest(
    repo_root: Path,
    index_target: Path,
    checksum_target: Path,
    indexed_files: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    if not checksum_target.is_file():
        errors.append(f"missing checksum manifest: {checksum_target}")
        return
    observed: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"invalid checksum line {line_number}")
            continue
        digest, relative_path = parts
        if relative_path in observed:
            errors.append(f"duplicate checksum path: {relative_path}")
        observed[relative_path] = digest

    expected = {
        relative_path: str(row["sha256"])
        for relative_path, row in indexed_files.items()
    }
    expected[index_target.relative_to(repo_root).as_posix()] = sha256_file(index_target)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            path
            for path in set(expected) & set(observed)
            if expected[path] != observed[path]
        )
        if missing:
            errors.append(f"checksum manifest missing paths: {missing}")
        if extra:
            errors.append(f"checksum manifest extra paths: {extra}")
        if mismatched:
            errors.append(f"checksum manifest digest mismatch: {mismatched}")


def _validate_p0_archived_inputs(repo_root: Path, errors: list[str]) -> None:
    protocol = _load_json(repo_root / "experiments/p0_p1_archive_verification_v1_protocol.json")
    archived_case = repo_root / protocol["source"]["archived_case_log"]
    archived_run = repo_root / protocol["source"]["archived_source_run"]
    if sha256_file(archived_case) != protocol["case_corpus"]["case_log_sha256"]:
        errors.append("P0 archived case-log checksum mismatch")
    if sha256_file(archived_run) != protocol["case_corpus"]["source_run_sha256"]:
        errors.append("P0 archived source-run checksum mismatch")


def _validate_p2_frozen_source_archive(repo_root: Path, errors: list[str]) -> None:
    archive_root = repo_root / P2_FROZEN_SOURCE_ROOT
    archive_manifest_path = archive_root / "archive_manifest.json"
    implementation_path = repo_root / "experiments/ccs_obligation_priority_v2_implementation.json"
    if not archive_manifest_path.is_file():
        errors.append("P2 frozen-source archive manifest is missing")
        return
    archive_manifest = _load_json(archive_manifest_path)
    implementation = _load_json(implementation_path)
    implementation_ref = archive_manifest.get("source_implementation_manifest", {})
    if implementation_ref.get("sha256") != sha256_file(implementation_path):
        errors.append("P2 frozen-source implementation-manifest checksum mismatch")
    archived_rows = {
        str(row.get("path", "")): row
        for row in archive_manifest.get("files", [])
        if isinstance(row, Mapping)
    }
    expected = implementation.get("source_sha256", {})
    if set(archived_rows) != set(expected):
        errors.append("P2 frozen-source archive path set mismatch")
        return
    for relative_path, expected_digest in expected.items():
        archived_path = archive_root / relative_path
        row = archived_rows[relative_path]
        if not archived_path.is_file():
            errors.append(f"P2 frozen-source file is missing: {relative_path}")
            continue
        if sha256_file(archived_path) != expected_digest:
            errors.append(f"P2 frozen-source checksum mismatch: {relative_path}")
        if int(row.get("size_bytes", -1)) != archived_path.stat().st_size:
            errors.append(f"P2 frozen-source size mismatch: {relative_path}")


def _validate_rlcmf_reanalysis(repo_root: Path, errors: list[str]) -> None:
    base = repo_root / "experiments/publication_baseline/v1/rlcmf_large96_v1"
    original = _load_json(base / "analysis.json")
    recomputed = _load_json(base / "analysis_recomputed.json")
    original_without_inputs = dict(original)
    recomputed_without_inputs = dict(recomputed)
    original_without_inputs.pop("inputs", None)
    recomputed_without_inputs.pop("inputs", None)
    if original_without_inputs != recomputed_without_inputs:
        errors.append("RLCMF recomputed analysis differs beyond repository input paths")
    if (base / "report.md").read_bytes() != (base / "report_recomputed.md").read_bytes():
        errors.append("RLCMF recomputed Markdown differs from archived original")
    for payload in (original, recomputed):
        inputs = payload.get("inputs", {})
        if inputs.get("manifest_file_sha256") != sha256_file(base / "manifest.json"):
            errors.append("RLCMF analysis manifest checksum mismatch")
        if inputs.get("result_file_sha256") != sha256_file(base / "result.json"):
            errors.append("RLCMF analysis result checksum mismatch")


def _require_repository_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"evidence path is not repository-relative: {value}")


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum_entry_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
