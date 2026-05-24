from pathlib import Path

from datadiff import final_readiness
from datadiff.final_readiness import (
    DEFAULT_A_LEVEL_READINESS_POLICY,
    ReadinessPolicy,
    ReadinessThresholds,
    build_final_readiness,
)
from datadiff.targets import TARGET_SUITES, describe_targets
from datadiff.util import append_jsonl, dump_json, run_meta_path


def test_final_readiness_passes_when_all_evidence_tracks_are_present(tmp_path):
    manifests = []
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
        )
    )

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(min_live_duration_hours=0.0),
    )

    assert audit["ready"] is True
    assert tuple(audit["policy"]["required_live_suites"]) == DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is True
    assert audit["summary"]["confirmed_live_candidate_families"] == {"confirmed_root@datafusion": 1}
    assert audit["summary"]["historical_confirmed_bug_ids"] == ["duckdb-22075", "duckdb-22656"]


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
    assert gates["live_suite_breadth"]["passed"] is False
    assert gates["latest_confirmed_bug_families"]["passed"] is False
    assert gates["historical_confirmed_replay"]["passed"] is False
    assert gates["seeded_sensitivity"]["passed"] is False


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
            require_seeded=False,
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
            require_seeded=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["ignored_evidence_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["gates"][0]["name"] == "live_suite_breadth"
    assert audit["gates"][0]["passed"] is False


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
            require_seeded=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["explicit_live_runs"] == 1
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["replay_policy_rejected_live_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["gates"][0]["name"] == "live_suite_breadth"
    assert audit["gates"][0]["passed"] is False


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
            require_seeded=False,
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
            require_seeded=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["known_saturated_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["confirmed_live_candidate_families"] == {}


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
            require_seeded=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("dataframe", "embedded_sql", "query_engine"),
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
            require_seeded=False,
        ),
    )

    assert md_path.exists()
    assert json_path.exists()
    assert "Final Experiment Readiness" in md_path.read_text(encoding="utf-8")


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
) -> Path:
    runs_dir = root / "runs"
    run_file = runs_dir / f"run-{name}.jsonl"
    backends = list(TARGET_SUITES[target_suite])
    append_jsonl(
        {
            "case_index": 0,
            "case": {"case_id": f"case-{name}", "seed": seed, "program": {"operations": []}},
            "findings": findings or [],
        },
        run_file,
    )
    dump_json(
        {
            "executed_cases": 1,
            "elapsed_s": 1.0,
            "throughput_cases_s": 1.0,
            "backends": backends,
            "targets": describe_targets(backends),
            "config": config or {},
            "replay_bug_filter": replay_filter or {},
        },
        run_meta_path(run_file),
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
    dump_json(manifest_payload, manifest)
    return manifest
