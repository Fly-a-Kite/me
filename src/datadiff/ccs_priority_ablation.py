from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.experiment_manifest import stable_digest
from datadiff.method_arms import arm_difference, method_arm
from datadiff.util import utc_now


CCS_PRIORITY_WORKER_SCHEMA_VERSION = "ccs-obligation-priority-worker-v1"
CCS_PRIORITY_RESULT_SCHEMA_VERSION = "ccs-obligation-priority-ablation-v1"
CCS_PRIORITY_BASE_OUTCOME_SCHEMA_VERSION = "ccs-priority-base-outcome-v2"
CCS_PRIORITY_ARMS = (
    "contract_ccs_obligations_cartesian",
    "contract_ccs_risk_obligations_cartesian",
)


def base_outcome_digest(
    normalized: Mapping[str, Any],
    differential_findings: Sequence[Mapping[str, Any]],
) -> str:
    """Digest only base execution and differential-oracle state.

    The overall run status is intentionally excluded because it also reflects
    which metamorphic variants an arm selected. Including it makes a newly
    discovered metamorphic candidate look like a base-execution mismatch.
    """

    return stable_digest(
        "base-outcome",
        {
            "schema_version": CCS_PRIORITY_BASE_OUTCOME_SCHEMA_VERSION,
            "normalized": dict(normalized),
            "differential_findings": [dict(row) for row in differential_findings],
        },
    )


def priority_block_schedule(block_count: int) -> list[dict[str, Any]]:
    return [
        {
            "block_index": block_index,
            "process_order": list(
                CCS_PRIORITY_ARMS
                if block_index % 2 == 0
                else tuple(reversed(CCS_PRIORITY_ARMS))
            ),
        }
        for block_index in range(max(1, int(block_count)))
    ]


def validate_priority_worker(worker: Mapping[str, Any]) -> None:
    if worker.get("schema_version") != CCS_PRIORITY_WORKER_SCHEMA_VERSION:
        raise ValueError("unsupported CCS priority worker schema")
    arm_id = str(worker.get("arm_id", ""))
    if arm_id not in CCS_PRIORITY_ARMS:
        raise ValueError(f"unsupported CCS priority arm: {arm_id}")
    runs = worker.get("runs", ())
    if not isinstance(runs, list):
        raise TypeError("CCS priority worker runs must be a list")
    for run in runs:
        if not isinstance(run, Mapping):
            raise TypeError("CCS priority worker run must be a mapping")
        expected = stable_digest(
            "ccs-priority-run",
            {key: value for key, value in run.items() if key != "run_digest"},
        )
        if run.get("run_digest") != expected:
            raise ValueError("CCS priority run digest mismatch")
    expected_worker = stable_digest(
        "ccs-priority-worker",
        {key: value for key, value in worker.items() if key != "worker_digest"},
    )
    if worker.get("worker_digest") != expected_worker:
        raise ValueError("CCS priority worker digest mismatch")


def combine_priority_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    corpus_digest: str,
    backends: Sequence[str],
    variant_limit: int,
    thread_environment: Mapping[str, str],
    warmup_seed: int,
) -> dict[str, Any]:
    expected_difference = {
        "obligation_priority_mode": (
            "complete_builder_order",
            "ccs_risk_priority",
        )
    }
    if arm_difference(
        method_arm(CCS_PRIORITY_ARMS[0]),
        method_arm(CCS_PRIORITY_ARMS[1]),
    ) != expected_difference:
        raise ValueError("CCS priority arms do not differ by exactly one dimension")

    normalized_blocks: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    arm_totals = {
        arm_id: {
            "case_count": 0,
            "backend_calls": 0,
            "selected_variant_count": 0,
            "finding_count": 0,
            "candidate_keys": set(),
            "selected_relations": Counter(),
            "process_cpu_seconds": 0.0,
            "external_process_wall_ms": 0.0,
        }
        for arm_id in CCS_PRIORITY_ARMS
    }
    candidates: list[dict[str, Any]] = []
    worker_artifacts: list[dict[str, Any]] = []

    for block in blocks:
        block_index = int(block["block_index"])
        workers = block.get("workers", {})
        if not isinstance(workers, Mapping):
            raise TypeError("CCS priority block workers must be a mapping")
        for arm_id in CCS_PRIORITY_ARMS:
            worker = workers.get(arm_id)
            if not isinstance(worker, Mapping):
                raise ValueError(
                    f"CCS priority block {block_index} is missing {arm_id}"
                )
            validate_priority_worker(worker)
            if worker.get("arm_id") != arm_id:
                raise ValueError("CCS priority worker arm id mismatch")
            if int(worker.get("block_index", -1)) != block_index:
                raise ValueError("CCS priority worker block index mismatch")
            expected_priority_mode = method_arm(
                arm_id
            ).obligation_priority_mode
            if worker.get("priority_mode") != expected_priority_mode:
                raise ValueError("CCS priority worker mode mismatch")
            if worker.get("corpus_digest") != corpus_digest:
                raise ValueError("CCS priority worker corpus digest mismatch")
            if list(worker.get("backends", ())) != list(backends):
                raise ValueError("CCS priority worker backend order mismatch")
            if int(worker.get("variant_limit", -1)) != int(variant_limit):
                raise ValueError("CCS priority worker variant limit mismatch")

        control = workers[CCS_PRIORITY_ARMS[0]]
        treatment = workers[CCS_PRIORITY_ARMS[1]]
        if (
            int(control["case_start"]) != int(treatment["case_start"])
            or int(control["case_end_exclusive"])
            != int(treatment["case_end_exclusive"])
        ):
            raise ValueError("CCS priority worker block ranges do not match")
        block_pairs = _pair_runs(control["runs"], treatment["runs"])
        pairs.extend(block_pairs)
        external_wall = block.get("external_process_wall_ms", {})
        normalized_blocks.append(
            {
                "block_index": block_index,
                "process_order": list(block.get("process_order", ())),
                "case_start": int(control["case_start"]),
                "case_end_exclusive": int(control["case_end_exclusive"]),
                "pair_count": len(block_pairs),
                "validity_gates": _pair_gates(block_pairs),
                "workers": {
                    arm_id: {
                        "worker_digest": workers[arm_id]["worker_digest"],
                        "summary": workers[arm_id]["summary"],
                        "warmup": workers[arm_id]["warmup"],
                        "process_resources": workers[arm_id]["process_resources"],
                        "external_process_wall_ms": float(
                            external_wall.get(arm_id, 0.0) or 0.0
                        ),
                    }
                    for arm_id in CCS_PRIORITY_ARMS
                },
            }
        )

        for arm_id in CCS_PRIORITY_ARMS:
            worker = workers[arm_id]
            total = arm_totals[arm_id]
            total["case_count"] += len(worker["runs"])
            total["backend_calls"] += sum(
                int(run.get("backend_calls", 0) or 0)
                for run in worker["runs"]
            )
            total["selected_variant_count"] += sum(
                int(run.get("selected_variant_count", 0) or 0)
                for run in worker["runs"]
            )
            total["finding_count"] += sum(
                len(run.get("findings", ())) for run in worker["runs"]
            )
            total["process_cpu_seconds"] += _process_cpu_seconds(
                worker.get("process_resources", {})
            )
            total["external_process_wall_ms"] += float(
                external_wall.get(arm_id, 0.0) or 0.0
            )
            for run in worker["runs"]:
                total["selected_relations"].update(
                    str(item) for item in run.get("selected_relations", ())
                )
                for finding in run.get("findings", ()):
                    candidate = {
                        "arm_id": arm_id,
                        "block_index": block_index,
                        "case_index": int(run["case_index"]),
                        "case_id": str(run["case_id"]),
                        "seed": int(run["seed"]),
                        "selected_variants": list(
                            run.get("selected_variants", ())
                        ),
                        **dict(finding),
                    }
                    candidate_key = _candidate_key(candidate)
                    candidate["candidate_key"] = candidate_key
                    total["candidate_keys"].add(candidate_key)
                    candidates.append(candidate)

            worker_artifacts.append(
                {
                    "block_index": block_index,
                    "arm_id": arm_id,
                    "path": str(block.get("worker_paths", {}).get(arm_id, "")),
                    "worker_digest": worker["worker_digest"],
                    "external_process_wall_ms": float(
                        external_wall.get(arm_id, 0.0) or 0.0
                    ),
                }
            )

    control_keys = arm_totals[CCS_PRIORITY_ARMS[0]]["candidate_keys"]
    treatment_keys = arm_totals[CCS_PRIORITY_ARMS[1]]["candidate_keys"]
    summarized_arms = {
        arm_id: _summarize_arm(arm_totals[arm_id])
        for arm_id in CCS_PRIORITY_ARMS
    }
    payload = {
        "schema_version": CCS_PRIORITY_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_digest": str(corpus_digest),
        "backends": [str(backend) for backend in backends],
        "arm_ids": list(CCS_PRIORITY_ARMS),
        "single_changed_dimension": "obligation_priority_mode",
        "variant_limit": int(variant_limit),
        "thread_environment": dict(thread_environment),
        "warmup_seed": int(warmup_seed),
        "warmup_excluded": True,
        "block_count": len(normalized_blocks),
        "case_pair_count": len(pairs),
        "blocks": normalized_blocks,
        "pairs": pairs,
        "validity_gates": {
            **_pair_gates(pairs),
            "same_total_backend_call_count": (
                summarized_arms[CCS_PRIORITY_ARMS[0]]["backend_calls"]
                == summarized_arms[CCS_PRIORITY_ARMS[1]]["backend_calls"]
            ),
            "same_total_selected_variant_count": (
                summarized_arms[CCS_PRIORITY_ARMS[0]][
                    "selected_variant_count"
                ]
                == summarized_arms[CCS_PRIORITY_ARMS[1]][
                    "selected_variant_count"
                ]
            ),
            "all_warmups_executed": all(
                bool(
                    block["workers"][arm_id]["warmup"].get("executed")
                )
                for block in blocks
                for arm_id in CCS_PRIORITY_ARMS
            ),
            "all_warmups_excluded": all(
                bool(
                    block["workers"][arm_id]["warmup"].get(
                        "excluded_from_metrics"
                    )
                )
                for block in blocks
                for arm_id in CCS_PRIORITY_ARMS
            ),
            "all_warmup_seeds_match": all(
                int(
                    block["workers"][arm_id]["warmup"].get("seed", -1)
                )
                == int(warmup_seed)
                for block in blocks
                for arm_id in CCS_PRIORITY_ARMS
            ),
            "all_workers_complete": len(worker_artifacts)
            == len(normalized_blocks) * len(CCS_PRIORITY_ARMS),
        },
        "arms": summarized_arms,
        "candidate_comparison": {
            "shared_unique_candidate_count": len(control_keys & treatment_keys),
            "control_only_unique_candidate_count": len(
                control_keys - treatment_keys
            ),
            "treatment_only_unique_candidate_count": len(
                treatment_keys - control_keys
            ),
            "union_unique_candidate_count": len(control_keys | treatment_keys),
        },
        "candidates": candidates,
        "worker_artifacts": worker_artifacts,
        "stopping_rule": "complete_all_frozen_blocks_without_outcome_based_early_stopping",
    }
    payload["result_digest"] = stable_digest("ccs-priority-ablation", payload)
    return payload


def _pair_runs(
    control_runs: Sequence[Mapping[str, Any]],
    treatment_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    control_by_index = {int(run["case_index"]): run for run in control_runs}
    treatment_by_index = {int(run["case_index"]): run for run in treatment_runs}
    if control_by_index.keys() != treatment_by_index.keys():
        raise ValueError("CCS priority worker case indices do not match")
    pairs = []
    for case_index in sorted(control_by_index):
        control = control_by_index[case_index]
        treatment = treatment_by_index[case_index]
        if control.get("case_id") != treatment.get("case_id"):
            raise ValueError("CCS priority paired case ids do not match")
        control_variants = set(control.get("selected_variants", ()))
        treatment_variants = set(treatment.get("selected_variants", ()))
        union = control_variants | treatment_variants
        pairs.append(
            {
                "case_index": case_index,
                "case_id": control["case_id"],
                "base_outcome_equal": (
                    control.get("base_outcome_digest")
                    == treatment.get("base_outcome_digest")
                ),
                "comparison_block_equal": (
                    control.get("comparison_block_digest")
                    == treatment.get("comparison_block_digest")
                ),
                "backend_calls_equal": (
                    int(control.get("backend_calls", 0) or 0)
                    == int(treatment.get("backend_calls", 0) or 0)
                ),
                "selected_variant_count_equal": (
                    int(control.get("selected_variant_count", 0) or 0)
                    == int(treatment.get("selected_variant_count", 0) or 0)
                ),
                "control_finding_count": len(control.get("findings", ())),
                "treatment_finding_count": len(treatment.get("findings", ())),
                "selected_variant_overlap_count": len(
                    control_variants & treatment_variants
                ),
                "selected_variant_union_count": len(union),
                "selected_variant_jaccard": (
                    len(control_variants & treatment_variants) / len(union)
                    if union
                    else 1.0
                ),
            }
        )
    return pairs


def _pair_gates(pairs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "base_outcome_equivalence_rate": _rate(pairs, "base_outcome_equal"),
        "comparison_block_match_rate": _rate(
            pairs, "comparison_block_equal"
        ),
        "backend_call_match_rate": _rate(pairs, "backend_calls_equal"),
        "selected_variant_count_match_rate": _rate(
            pairs, "selected_variant_count_equal"
        ),
    }


def _summarize_arm(total: Mapping[str, Any]) -> dict[str, Any]:
    cpu_seconds = float(total["process_cpu_seconds"])
    case_count = int(total["case_count"])
    candidate_keys = set(total["candidate_keys"])
    return {
        "case_count": case_count,
        "backend_calls": int(total["backend_calls"]),
        "selected_variant_count": int(total["selected_variant_count"]),
        "finding_count": int(total["finding_count"]),
        "unique_candidate_count": len(candidate_keys),
        "process_cpu_seconds": cpu_seconds,
        "external_process_wall_ms": float(total["external_process_wall_ms"]),
        "cases_per_cpu_hour": (
            case_count * 3600.0 / cpu_seconds if cpu_seconds > 0.0 else None
        ),
        "selected_relation_counts": dict(
            sorted(Counter(total["selected_relations"]).items())
        ),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    signature = str(candidate.get("signature", "") or "")
    if signature:
        return signature
    return stable_digest(
        "candidate",
        {
            "case_id": candidate.get("case_id", ""),
            "kind": candidate.get("kind", ""),
            "root_cause": candidate.get("root_cause", ""),
            "mismatch_class": candidate.get("mismatch_class", ""),
            "suspicious_backends": candidate.get("suspicious_backends", ()),
        },
    )


def _process_cpu_seconds(resources: Mapping[str, Any]) -> float:
    return float(resources.get("user_cpu_seconds", 0.0) or 0.0) + float(
        resources.get("system_cpu_seconds", 0.0) or 0.0
    )


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return (
        sum(bool(row.get(key)) for row in rows) / len(rows)
        if rows
        else 1.0
    )
