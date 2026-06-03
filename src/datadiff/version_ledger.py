from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from datadiff.adaptive_learning import continual_learning_summary
from datadiff.reward import candidate_issue_family_key, is_candidate_issue_finding, is_false_positive_finding
from datadiff.util import dump_json, read_jsonl, run_meta_path, utc_now

VERSION_LEDGER_SCHEMA_VERSION = "version-ledger-v1"
OBSERVATION_PRESENT = "present"
OBSERVATION_ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class VersionObservation:
    version_id: str
    run_file: str
    case_count: int
    candidate_families: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)
    invalid_case_count: int = 0
    fallback_case_count: int = 0
    false_positive_count: int = 0
    duration_ms_total: float = 0.0
    throughput_cases_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "run_file": self.run_file,
            "case_count": self.case_count,
            "candidate_families": dict(sorted(self.candidate_families.items())),
            "health": observation_health_summary(self),
            "metadata": dict(self.metadata),
        }


def observation_from_run_log(
    run_file: str | Path,
    *,
    version_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> VersionObservation:
    path = Path(run_file)
    rows = read_jsonl(path)
    meta = _load_run_meta(path)
    resolved_version = (
        str(version_id).strip()
        or str(meta.get("target_version", "") or meta.get("version_id", "") or "").strip()
        or path.stem
    )
    family_counter: Counter[str] = Counter()
    invalid_case_count = 0
    fallback_case_count = 0
    false_positive_count = 0
    duration_ms_total = 0.0
    for row in rows:
        preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
        invalid_case_count += int(not bool(preflight.get("valid", True)))
        fallback_case_count += int(bool(preflight.get("fallback_used", False)))
        duration_ms_total += _float_value(row.get("duration_ms"))
        for finding in row.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            false_positive_count += int(is_false_positive_finding(finding))
            if is_candidate_issue_finding(finding):
                family_counter[candidate_issue_family_key(finding)] += 1
    merged_metadata = dict(meta)
    merged_metadata.update(dict(metadata or {}))
    meta_preflight = meta.get("preflight", {}) if isinstance(meta.get("preflight", {}), dict) else {}
    invalid_case_count = max(invalid_case_count, int(meta_preflight.get("invalid_cases", 0) or 0))
    fallback_case_count = max(fallback_case_count, int(meta_preflight.get("fallback_cases", 0) or 0))
    throughput_cases_s = _float_value(meta.get("throughput_cases_s"))
    if throughput_cases_s <= 0.0:
        elapsed_s = _float_value(meta.get("elapsed_s"))
        throughput_cases_s = len(rows) / elapsed_s if elapsed_s > 0.0 else 0.0
    return VersionObservation(
        version_id=resolved_version,
        run_file=str(path),
        case_count=len(rows),
        candidate_families=dict(family_counter),
        metadata=merged_metadata,
        invalid_case_count=invalid_case_count,
        fallback_case_count=fallback_case_count,
        false_positive_count=false_positive_count,
        duration_ms_total=duration_ms_total,
        throughput_cases_s=throughput_cases_s,
    )


def build_version_ledger(
    observations: Iterable[VersionObservation | Mapping[str, Any]],
    *,
    baseline_version: str = "",
    previous_ledger: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized = [_coerce_observation(item) for item in observations]
    normalized = [item for item in normalized if item.version_id]
    previous_presence = _previous_presence(previous_ledger)
    families = sorted({family for item in normalized for family in item.candidate_families})
    observation_map = {
        item.version_id: set(item.candidate_families)
        for item in normalized
    }
    version_order = [item.version_id for item in normalized]
    baseline = baseline_version or (version_order[0] if version_order else "")
    baseline_families = observation_map.get(baseline, set())
    family_rows = []
    transitions: Counter[str] = Counter()
    for family in families:
        states = [
            {
                "version_id": version,
                "state": OBSERVATION_PRESENT if family in observation_map.get(version, set()) else OBSERVATION_ABSENT,
                "count": _family_count(normalized, version, family),
            }
            for version in version_order
        ]
        first_seen = next((state["version_id"] for state in states if state["state"] == OBSERVATION_PRESENT), "")
        last_seen = next((state["version_id"] for state in reversed(states) if state["state"] == OBSERVATION_PRESENT), "")
        status = _family_status(
            family,
            states=states,
            baseline_families=baseline_families,
            previous_presence=previous_presence,
        )
        transitions[status] += 1
        family_rows.append(
            {
                "family": family,
                "status": status,
                "first_seen_version": first_seen,
                "last_seen_version": last_seen,
                "observations": states,
                "health_observations": [
                    _family_health_observation(item, family)
                    for item in normalized
                    if family in item.candidate_families
                ],
            }
        )
    continual_learning = continual_learning_summary(family_rows, version_order=version_order)
    health = ledger_health_summary(normalized)
    health_feedback_report = ledger_health_feedback_report(health)
    return {
        "schema_version": VERSION_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now(),
        "baseline_version": baseline,
        "version_order": version_order,
        "versions": [item.to_dict() for item in normalized],
        "families": family_rows,
        "summary": {
            "version_count": len(version_order),
            "family_count": len(family_rows),
            "transition_counts": dict(sorted(transitions.items())),
            "new_family_count": transitions["new"],
            "fixed_family_count": transitions["fixed"],
            "regression_family_count": transitions["regression"],
            "persistent_family_count": transitions["persistent"],
            "health_observation_count": health["health_observation_count"],
            "invalid_case_count": health["invalid_case_count"],
            "fallback_case_count": health["fallback_case_count"],
            "false_positive_count": health["false_positive_count"],
            "min_throughput_cases_s": health["min_throughput_cases_s"],
            "max_invalid_rate": health["max_invalid_rate"],
            "max_false_positive_rate": health["max_false_positive_rate"],
        },
        "health": health,
        "health_feedback_report": health_feedback_report,
        "continual_learning": continual_learning,
        "adaptive_learning_seed": continual_learning.get("adaptive_learning_seed", {}),
        "methodology_claim": (
            "The same semantic corpus can be replayed across versions to separate "
            "new bugs, fixes, regressions, persistent known failures, and unhealthy "
            "exploration regions that should be downweighted in future runs."
        ),
    }


def write_version_ledger(
    observations: Iterable[VersionObservation | Mapping[str, Any]],
    output_path: str | Path,
    *,
    baseline_version: str = "",
    previous_ledger: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    payload = build_version_ledger(
        observations,
        baseline_version=baseline_version,
        previous_ledger=previous_ledger,
    )
    dump_json(payload, path)
    return path


def observations_from_run_logs(
    run_files: Iterable[str | Path],
    *,
    versions: Iterable[str] | None = None,
) -> list[VersionObservation]:
    version_list = [str(version).strip() for version in versions or []]
    observations: list[VersionObservation] = []
    for index, run_file in enumerate(run_files):
        version_id = version_list[index] if index < len(version_list) else ""
        observations.append(observation_from_run_log(run_file, version_id=version_id))
    return observations


def _coerce_observation(value: VersionObservation | Mapping[str, Any]) -> VersionObservation:
    if isinstance(value, VersionObservation):
        return value
    health = value.get("health", {}) if isinstance(value.get("health", {}), Mapping) else {}
    return VersionObservation(
        version_id=str(value.get("version_id", "")).strip(),
        run_file=str(value.get("run_file", "")).strip(),
        case_count=int(value.get("case_count", 0) or 0),
        candidate_families={
            str(key): int(count or 0)
            for key, count in (value.get("candidate_families", {}) or {}).items()
            if str(key).strip() and int(count or 0) > 0
        },
        metadata=dict(value.get("metadata", {}) or {}),
        invalid_case_count=_int_health_field(value, health, "invalid_case_count"),
        fallback_case_count=_int_health_field(value, health, "fallback_case_count"),
        false_positive_count=_int_health_field(value, health, "false_positive_count"),
        duration_ms_total=_float_health_field(value, health, "duration_ms_total"),
        throughput_cases_s=_float_health_field(value, health, "throughput_cases_s"),
    )


def observation_health_summary(observation: VersionObservation) -> dict[str, Any]:
    cases = max(1, int(observation.case_count))
    invalid = max(0, int(observation.invalid_case_count))
    fallback = max(0, int(observation.fallback_case_count))
    false_positive = max(0, int(observation.false_positive_count))
    duration_ms_total = max(0.0, float(observation.duration_ms_total))
    throughput = max(0.0, float(observation.throughput_cases_s))
    return {
        "schema_version": "version-observation-health-v1",
        "has_health_feedback": _observation_has_health_feedback(observation),
        "invalid_case_count": invalid,
        "fallback_case_count": fallback,
        "false_positive_count": false_positive,
        "duration_ms_total": duration_ms_total,
        "throughput_cases_s": throughput,
        "invalid_rate": invalid / cases,
        "fallback_rate": fallback / cases,
        "false_positive_rate": false_positive / cases,
        "runtime_ms_per_case": duration_ms_total / cases,
    }


def ledger_health_summary(observations: Iterable[VersionObservation]) -> dict[str, Any]:
    rows = [observation_health_summary(observation) for observation in observations]
    health_rows = [row for row in rows if bool(row.get("has_health_feedback"))]
    throughputs = [float(row["throughput_cases_s"]) for row in health_rows if float(row["throughput_cases_s"]) > 0.0]
    runtime_rows = [float(row["runtime_ms_per_case"]) for row in health_rows if float(row["runtime_ms_per_case"]) > 0.0]
    return {
        "schema_version": "version-ledger-health-v1",
        "observation_count": len(rows),
        "health_observation_count": len(health_rows),
        "invalid_case_count": sum(int(row["invalid_case_count"]) for row in rows),
        "fallback_case_count": sum(int(row["fallback_case_count"]) for row in rows),
        "false_positive_count": sum(int(row["false_positive_count"]) for row in rows),
        "duration_ms_total": sum(float(row["duration_ms_total"]) for row in rows),
        "min_throughput_cases_s": min(throughputs) if throughputs else 0.0,
        "avg_throughput_cases_s": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "max_runtime_ms_per_case": max(runtime_rows) if runtime_rows else 0.0,
        "max_invalid_rate": max((float(row["invalid_rate"]) for row in rows), default=0.0),
        "max_fallback_rate": max((float(row["fallback_rate"]) for row in rows), default=0.0),
        "max_false_positive_rate": max((float(row["false_positive_rate"]) for row in rows), default=0.0),
        "observations": rows,
    }


def ledger_health_feedback_report(health: Mapping[str, Any]) -> dict[str, Any]:
    observation_count = max(0, int(_float_value(health.get("observation_count"))))
    health_observation_count = max(0, int(_float_value(health.get("health_observation_count"))))
    return {
        "schema_version": "version-ledger-health-feedback-report-v1",
        "has_health_feedback": health_observation_count > 0,
        "observation_count": observation_count,
        "health_observation_count": health_observation_count,
        "health_observation_coverage_rate": (
            health_observation_count / observation_count if observation_count else 0.0
        ),
        "invalid_case_count": max(0, int(_float_value(health.get("invalid_case_count")))),
        "fallback_case_count": max(0, int(_float_value(health.get("fallback_case_count")))),
        "false_positive_count": max(0, int(_float_value(health.get("false_positive_count")))),
        "duration_ms_total": max(0.0, _float_value(health.get("duration_ms_total"))),
        "min_throughput_cases_s": max(0.0, _float_value(health.get("min_throughput_cases_s"))),
        "avg_throughput_cases_s": max(0.0, _float_value(health.get("avg_throughput_cases_s"))),
        "max_runtime_ms_per_case": max(0.0, _float_value(health.get("max_runtime_ms_per_case"))),
        "max_invalid_rate": max(0.0, _float_value(health.get("max_invalid_rate"))),
        "max_fallback_rate": max(0.0, _float_value(health.get("max_fallback_rate"))),
        "max_false_positive_rate": max(0.0, _float_value(health.get("max_false_positive_rate"))),
    }


def _family_health_observation(observation: VersionObservation, family: str) -> dict[str, Any]:
    health = observation_health_summary(observation)
    return {
        "version_id": observation.version_id,
        "family_count": int(observation.candidate_families.get(family, 0) or 0),
        "invalid_rate": health["invalid_rate"],
        "fallback_rate": health["fallback_rate"],
        "false_positive_rate": health["false_positive_rate"],
        "runtime_ms_per_case": health["runtime_ms_per_case"],
        "throughput_cases_s": health["throughput_cases_s"],
    }


def _observation_has_health_feedback(observation: VersionObservation) -> bool:
    return bool(
        observation.invalid_case_count
        or observation.fallback_case_count
        or observation.false_positive_count
        or observation.duration_ms_total
        or observation.throughput_cases_s
    )


def _int_health_field(value: Mapping[str, Any], health: Mapping[str, Any], key: str) -> int:
    return max(0, int(_float_health_field(value, health, key)))


def _float_health_field(value: Mapping[str, Any], health: Mapping[str, Any], key: str) -> float:
    direct = _float_value(value.get(key))
    if direct:
        return direct
    return _float_value(health.get(key))


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_run_meta(run_file: Path) -> dict[str, Any]:
    meta_path = run_meta_path(run_file)
    if not meta_path.is_file():
        return {}
    try:
        import json

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _family_count(observations: list[VersionObservation], version: str, family: str) -> int:
    for observation in observations:
        if observation.version_id == version:
            return int(observation.candidate_families.get(family, 0) or 0)
    return 0


def _family_status(
    family: str,
    *,
    states: list[dict[str, Any]],
    baseline_families: set[str],
    previous_presence: dict[str, set[str]],
) -> str:
    present_versions = [state["version_id"] for state in states if state["state"] == OBSERVATION_PRESENT]
    if not present_versions:
        return "absent"
    first_present_index = next(
        index for index, state in enumerate(states) if state["state"] == OBSERVATION_PRESENT
    )
    last_state = states[-1]["state"] if states else OBSERVATION_ABSENT
    was_previously_present = any(family in values for values in previous_presence.values())
    if family not in baseline_families and first_present_index > 0:
        if was_previously_present:
            return "regression"
        return "new"
    if family in baseline_families and last_state == OBSERVATION_ABSENT:
        return "fixed"
    if len(present_versions) >= 2:
        return "persistent"
    return "new" if family not in baseline_families else "persistent"


def _previous_presence(previous_ledger: Mapping[str, Any] | None) -> dict[str, set[str]]:
    presence: dict[str, set[str]] = defaultdict(set)
    if not isinstance(previous_ledger, Mapping):
        return presence
    for row in previous_ledger.get("families", []) or []:
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("family", "")).strip()
        if not family:
            continue
        for state in row.get("observations", []) or []:
            if not isinstance(state, Mapping):
                continue
            if str(state.get("state", "")) == OBSERVATION_PRESENT:
                presence[str(state.get("version_id", "")).strip()].add(family)
    return presence
