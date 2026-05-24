from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datadiff.historical import list_historical_bugs
from datadiff.reward import (
    candidate_bug_family_key,
    is_issue_replay_finding,
    is_known_saturated_candidate_bug_finding,
    is_rewardable_candidate_bug_finding,
)
from datadiff.targets import TARGETS, TARGET_SUITES
from datadiff.util import REPORTS_DIR, RUNS_DIR, dump_json, ensure_dirs, load_json, read_jsonl, run_meta_path, utc_now


# Layering: the audit engine below is middle-layer analysis over manifests and
# run logs. A-level requirements are supplied as policy data and never alter the
# bottom-layer generator, runner, oracle, normalizer, or backend adapters.
@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    required_live_suites: tuple[str, ...] = (
        "datafusion_cross",
        "dataframe_lazy",
        "arrow_cross",
        "embedded_sql",
        "latest_all_engines",
        "latest_no_datafusion",
    )
    required_live_families: tuple[str, ...] = ("arrow", "dataframe", "embedded_sql", "query_engine")
    confirmed_live_paper_statuses: tuple[str, ...] = (
        "confirmed_bug",
        "confirmed_implementation_bug",
        "fixed_upstream",
        "maintainer_confirmed_bug",
    )


DEFAULT_A_LEVEL_READINESS_POLICY = ReadinessPolicy()
EVIDENCE_MODES = {"live", "historical", "seeded"}


@dataclass(frozen=True, slots=True)
class ReadinessThresholds:
    min_live_cases_per_suite: int = 1
    min_live_duration_hours: float = 24.0
    min_live_candidate_families: int = 1
    min_confirmed_live_families: int = 1
    min_historical_confirmed: int = 2
    require_seeded: bool = True


def analyze_final_readiness(
    manifest_files: list[Path] | None = None,
    *,
    thresholds: ReadinessThresholds | None = None,
    policy: ReadinessPolicy | None = None,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or ReadinessThresholds()
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    manifest_files = _resolve_manifest_files(manifest_files)
    audit = build_final_readiness(manifest_files, thresholds=thresholds, policy=policy)
    stamp = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    md_path = REPORTS_DIR / f"final-readiness-{stamp}.md"
    json_path = REPORTS_DIR / f"final-readiness-{stamp}.json"
    dump_json(audit, json_path)
    md_path.write_text(_render_markdown(audit), encoding="utf-8")
    return md_path, json_path


def build_final_readiness(
    manifest_files: list[Path],
    *,
    thresholds: ReadinessThresholds,
    policy: ReadinessPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    manifests = [_load_manifest(path) for path in manifest_files]
    runs = [run for manifest in manifests for run in _manifest_runs(manifest, policy=policy)]
    explicit_live_runs = [run for run in runs if run["evidence_mode"] == "live"]
    live_runs = [run for run in explicit_live_runs if _fresh_replay_policy_ok(run)]
    replay_policy_rejected_live_runs = [run for run in explicit_live_runs if not _fresh_replay_policy_ok(run)]
    historical_runs = [run for run in runs if run["evidence_mode"] == "historical"]
    seeded_runs = [run for run in runs if run["evidence_mode"] == "seeded"]
    ignored_runs = [run for run in runs if run["evidence_mode"] not in EVIDENCE_MODES]

    live_by_suite = _live_suite_summary(live_runs)
    live_families = sorted({family for run in live_runs for family in run["target_families"]})
    rewardable_live_families = Counter()
    confirmed_live_families = Counter()
    issue_replay_live_families = Counter()
    known_saturated_live_families = Counter()
    for run in live_runs:
        rewardable_live_families.update(run["rewardable_candidate_families"])
        confirmed_live_families.update(run["confirmed_candidate_families"])
        issue_replay_live_families.update(run["issue_replay_candidate_families"])
        known_saturated_live_families.update(run["known_saturated_candidate_families"])

    historical_confirmed = _historical_confirmed_runs(historical_runs)
    gates = _readiness_gates(
        thresholds=thresholds,
        live_by_suite=live_by_suite,
        live_families=live_families,
        live_runs=live_runs,
        historical_confirmed=historical_confirmed,
        seeded_runs=seeded_runs,
        rewardable_live_families=rewardable_live_families,
        confirmed_live_families=confirmed_live_families,
        policy=policy,
    )
    return {
        "created_at": utc_now(),
        "ready": all(gate["passed"] for gate in gates),
        "thresholds": asdict(thresholds),
        "policy": asdict(policy),
        "manifest_files": [str(path) for path in manifest_files],
        "gates": gates,
        "summary": {
            "live_runs": len(live_runs),
            "explicit_live_runs": len(explicit_live_runs),
            "replay_policy_rejected_live_runs": len(replay_policy_rejected_live_runs),
            "historical_runs": len(historical_runs),
            "seeded_runs": len(seeded_runs),
            "ignored_evidence_runs": len(ignored_runs),
            "live_suites": sorted(live_by_suite),
            "live_families": live_families,
            "rewardable_live_candidate_families": dict(sorted(rewardable_live_families.items())),
            "confirmed_live_candidate_families": dict(sorted(confirmed_live_families.items())),
            "issue_replay_live_candidate_families": dict(sorted(issue_replay_live_families.items())),
            "known_saturated_live_candidate_families": dict(sorted(known_saturated_live_families.items())),
            "historical_confirmed_bug_ids": sorted(historical_confirmed),
            "total_live_cases": sum(item["cases"] for item in live_by_suite.values()),
            "total_live_elapsed_s": sum(item["elapsed_s"] for item in live_by_suite.values()),
        },
        "live_suites": live_by_suite,
        "runs": live_runs + historical_runs + seeded_runs,
        "ignored_runs": ignored_runs,
        "replay_policy_rejected_live_runs": replay_policy_rejected_live_runs,
    }


def _resolve_manifest_files(manifest_files: list[Path] | None) -> list[Path]:
    if manifest_files:
        return [Path(path) for path in manifest_files]
    return sorted(RUNS_DIR.glob("experiment-*.json"))


def _load_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    data["_manifest_file"] = str(path)
    return data


def _manifest_runs(manifest: dict[str, Any], *, policy: ReadinessPolicy) -> list[dict[str, Any]]:
    manifest_mode = _normalize_evidence_mode(manifest.get("evidence_mode", ""))
    manifest_replay_policy = (
        manifest.get("replay_bug_policy", {}) if isinstance(manifest.get("replay_bug_policy", {}), dict) else {}
    )
    out = []
    for run in manifest.get("runs", []):
        run_file = Path(run.get("run_file", ""))
        rows = read_jsonl(run_file) if run_file.is_file() else []
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.is_file() else {}
        config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        known_bug_families = list(config.get("known_saturated_bug_families", []) or [])
        replay_filter = (
            meta.get("replay_bug_filter", {}) if isinstance(meta.get("replay_bug_filter", {}), dict) else {}
        )
        target_suite = str(run.get("target_suite", manifest.get("target_suite", "")) or "")
        target_families = _target_families(run, manifest, meta)
        evidence_mode = _normalize_evidence_mode(run.get("evidence_mode", manifest_mode))
        if evidence_mode == "live" and _has_seeded_target(target_suite, target_families):
            evidence_mode = "unknown"
        run_enable_replay_bug = (
            bool(config.get("enable_replay_bug"))
            if "enable_replay_bug" in config
            else evidence_mode == "historical"
        )
        manifest_enable_replay_bug = (
            bool(manifest_replay_policy.get("enable_replay_bug"))
            if "enable_replay_bug" in manifest_replay_policy
            else evidence_mode == "historical"
        )
        findings = [finding for row in rows for finding in row.get("findings", [])]
        out.append(
            {
                "manifest_file": manifest.get("_manifest_file", ""),
                "run_file": str(run_file),
                "target_suite": target_suite,
                "preset": str(run.get("preset", "")),
                "seed": run.get("seed", ""),
                "evidence_mode": evidence_mode,
                "known_bug_id": str(run.get("known_bug_id", manifest.get("known_bug_id", "")) or ""),
                "target_version": str(run.get("target_version", manifest.get("target_version", "")) or ""),
                "cases": int(meta.get("executed_cases", len(rows)) or 0),
                "elapsed_s": float(meta.get("elapsed_s", 0.0) or 0.0),
                "duration_s": meta.get("duration_s"),
                "findings": len(findings),
                "target_families": target_families,
                "enable_replay_bug": run_enable_replay_bug,
                "manifest_enable_replay_bug": manifest_enable_replay_bug,
                "replay_filter_enabled": replay_filter.get("enabled", ""),
                "replay_filter_filtered_candidates": int(replay_filter.get("filtered_candidates", 0) or 0),
                "known_saturated_bug_families": known_bug_families,
                "rewardable_candidate_families": _candidate_family_counter(
                    findings,
                    rewardable=True,
                    known_saturated_bug_families=known_bug_families,
                ),
                "confirmed_candidate_families": _candidate_family_counter(
                    findings,
                    confirmed=True,
                    known_saturated_bug_families=known_bug_families,
                    policy=policy,
                ),
                "issue_replay_candidate_families": _candidate_family_counter(findings, issue_replay=True),
                "known_saturated_candidate_families": _candidate_family_counter(
                    findings,
                    known_saturated=True,
                    known_saturated_bug_families=known_bug_families,
                ),
            }
        )
    return out


def _normalize_evidence_mode(value: Any) -> str:
    mode = str(value or "").strip()
    return mode if mode in EVIDENCE_MODES else "unknown"


def _has_seeded_target(target_suite: str, target_families: list[str]) -> bool:
    return target_suite.startswith("seeded_") or "seeded_fault" in set(target_families)


def _target_families(run: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    targets = meta.get("targets") or manifest.get("targets") or []
    families = {str(target.get("family", "")) for target in targets if isinstance(target, dict)}
    if families:
        return sorted(family for family in families if family)
    backends = run.get("backends") or manifest.get("backends_by_suite", {}).get(run.get("target_suite"), [])
    if not backends:
        backends = TARGET_SUITES.get(str(run.get("target_suite", "")), [])
    return sorted({TARGETS[backend].family for backend in backends if backend in TARGETS})


def _candidate_family_counter(
    findings: list[dict[str, Any]],
    *,
    rewardable: bool = False,
    confirmed: bool = False,
    issue_replay: bool = False,
    known_saturated: bool = False,
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    policy: ReadinessPolicy | None = None,
) -> Counter[str]:
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    confirmed_statuses = set(policy.confirmed_live_paper_statuses)
    counter: Counter[str] = Counter()
    for finding in findings:
        if str(finding.get("triage_verdict", "")) != "candidate_implementation_bug":
            continue
        if rewardable and not is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families):
            continue
        if confirmed and str(finding.get("paper_status", "")) not in confirmed_statuses:
            continue
        if confirmed and is_known_saturated_candidate_bug_finding(finding, known_saturated_bug_families):
            continue
        if issue_replay and not is_issue_replay_finding(finding):
            continue
        if known_saturated and not is_known_saturated_candidate_bug_finding(
            finding,
            known_saturated_bug_families,
        ):
            continue
        counter[_candidate_family_key(finding)] += 1
    return counter


def _candidate_family_key(finding: dict[str, Any]) -> str:
    return candidate_bug_family_key(finding)


def _live_suite_summary(live_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_suite: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "cases": 0,
            "elapsed_s": 0.0,
            "findings": 0,
            "seeds": [],
            "presets": [],
            "families": set(),
            "replay_policy_ok": True,
        }
    )
    for run in live_runs:
        item = by_suite[run["target_suite"]]
        item["runs"] += 1
        item["cases"] += int(run["cases"])
        item["elapsed_s"] += float(run["elapsed_s"])
        item["findings"] += int(run["findings"])
        item["seeds"].append(run["seed"])
        item["presets"].append(run["preset"])
        item["families"].update(run["target_families"])
        item["replay_policy_ok"] = item["replay_policy_ok"] and _fresh_replay_policy_ok(run)
    return {
        suite: {
            **item,
            "seeds": sorted({str(seed) for seed in item["seeds"]}),
            "presets": sorted({str(preset) for preset in item["presets"]}),
            "families": sorted(item["families"]),
        }
        for suite, item in sorted(by_suite.items())
    }


def _fresh_replay_policy_ok(run: dict[str, Any]) -> bool:
    return (
        run["evidence_mode"] == "live"
        and run["enable_replay_bug"] is False
        and run["manifest_enable_replay_bug"] is False
        and run["replay_filter_enabled"] is True
    )


def _historical_confirmed_runs(historical_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    confirmed_ids = {spec.bug_id for spec in list_historical_bugs(include_pending=False)}
    out = {}
    for run in historical_runs:
        bug_id = run["known_bug_id"]
        if bug_id in confirmed_ids and run["enable_replay_bug"] is True:
            out[bug_id] = run
    return out


def _readiness_gates(
    *,
    thresholds: ReadinessThresholds,
    live_by_suite: dict[str, dict[str, Any]],
    live_families: list[str],
    live_runs: list[dict[str, Any]],
    historical_confirmed: dict[str, dict[str, Any]],
    seeded_runs: list[dict[str, Any]],
    rewardable_live_families: Counter[str],
    confirmed_live_families: Counter[str],
    policy: ReadinessPolicy,
) -> list[dict[str, Any]]:
    required_suites = set(policy.required_live_suites)
    missing_suites = sorted(required_suites - set(live_by_suite))
    shallow_suites = sorted(
        suite
        for suite in required_suites & set(live_by_suite)
        if live_by_suite[suite]["cases"] < thresholds.min_live_cases_per_suite
        or live_by_suite[suite]["elapsed_s"] < thresholds.min_live_duration_hours * 3600
    )
    missing_families = sorted(set(policy.required_live_families) - set(live_families))
    replay_policy_bad = sorted({run["target_suite"] for run in live_runs if not _fresh_replay_policy_ok(run)})
    return [
        _gate(
            "live_suite_breadth",
            not missing_suites,
            f"required={len(policy.required_live_suites)} covered={len(required_suites - set(missing_suites))}",
            missing=missing_suites,
        ),
        _gate(
            "live_target_family_breadth",
            not missing_families,
            f"required={','.join(policy.required_live_families)} covered={','.join(live_families) or 'none'}",
            missing=missing_families,
        ),
        _gate(
            "live_depth",
            not shallow_suites and not missing_suites,
            (
                f"min_cases_per_suite={thresholds.min_live_cases_per_suite} "
                f"min_elapsed_hours={thresholds.min_live_duration_hours:g}"
            ),
            missing=missing_suites,
            shallow=shallow_suites,
        ),
        _gate(
            "fresh_replay_policy",
            not replay_policy_bad and bool(live_runs),
            "live runs must keep enable_replay_bug=false and replay filter enabled",
            bad_suites=replay_policy_bad,
        ),
        _gate(
            "latest_candidate_bug_families",
            len(rewardable_live_families) >= thresholds.min_live_candidate_families,
            f"required={thresholds.min_live_candidate_families} observed={len(rewardable_live_families)}",
            families=sorted(rewardable_live_families),
        ),
        _gate(
            "latest_confirmed_bug_families",
            len(confirmed_live_families) >= thresholds.min_confirmed_live_families,
            f"required={thresholds.min_confirmed_live_families} observed={len(confirmed_live_families)}",
            families=sorted(confirmed_live_families),
        ),
        _gate(
            "historical_confirmed_replay",
            len(historical_confirmed) >= thresholds.min_historical_confirmed,
            f"required={thresholds.min_historical_confirmed} observed={len(historical_confirmed)}",
            bug_ids=sorted(historical_confirmed),
        ),
        _gate(
            "seeded_sensitivity",
            bool(seeded_runs) or not thresholds.require_seeded,
            f"required={thresholds.require_seeded} observed_runs={len(seeded_runs)}",
        ),
    ]


def _gate(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, **extra}


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Final Experiment Readiness",
        "",
        f"- Created at: `{audit['created_at']}`",
        f"- Ready: `{str(audit['ready']).lower()}`",
        f"- Manifests: `{len(audit['manifest_files'])}`",
        "",
        "## Gates",
        "",
        "| gate | status | detail |",
        "|---|---|---|",
    ]
    for gate in audit["gates"]:
        lines.append(f"| {gate['name']} | {'pass' if gate['passed'] else 'fail'} | {gate['detail']} |")
    summary = audit["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Live runs: `{summary['live_runs']}`; live cases: `{summary['total_live_cases']}`; live elapsed_s: `{summary['total_live_elapsed_s']:.3f}`",
            f"- Explicit live runs: `{summary['explicit_live_runs']}`; replay-policy rejected live runs: `{summary['replay_policy_rejected_live_runs']}`",
            f"- Ignored evidence runs: `{summary['ignored_evidence_runs']}`",
            f"- Live suites: `{', '.join(summary['live_suites']) or 'none'}`",
            f"- Live families: `{', '.join(summary['live_families']) or 'none'}`",
            f"- Rewardable latest candidate families: `{len(summary['rewardable_live_candidate_families'])}`",
            f"- Known/saturated latest candidate families: `{len(summary['known_saturated_live_candidate_families'])}`",
            f"- Confirmed latest candidate families: `{len(summary['confirmed_live_candidate_families'])}`",
            f"- Historical confirmed replay ids: `{', '.join(summary['historical_confirmed_bug_ids']) or 'none'}`",
            "",
            "## Live Suites",
            "",
            "| suite | runs | cases | elapsed_s | findings | families | replay policy |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for suite, row in audit["live_suites"].items():
        lines.append(
            f"| {suite} | {row['runs']} | {row['cases']} | {row['elapsed_s']:.3f} | "
            f"{row['findings']} | {', '.join(row['families'])} | {'ok' if row['replay_policy_ok'] else 'bad'} |"
        )
    lines.append("")
    return "\n".join(lines)
