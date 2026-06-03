from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

HistoricalBugStatus = Literal["confirmed_fixed", "pending_merge", "candidate"]
HistoricalReplayKind = Literal["experiment", "fixture"]


@dataclass(frozen=True, slots=True)
class HistoricalBugSpec:
    bug_id: str
    project: str
    status: HistoricalBugStatus
    target_suite: str
    target_version: str
    default_presets: tuple[str, ...]
    default_cases: int
    default_seeds: tuple[int, ...]
    expected_root_causes: tuple[str, ...] = field(default_factory=tuple)
    expected_suspicious_backends: tuple[str, ...] = field(default_factory=tuple)
    fixed_version: str = ""
    issue_url: str = ""
    notes: str = ""
    replay_kind: HistoricalReplayKind = "experiment"
    fixture_spec: str = ""
    fixture_env: str = ""
    default_artifact_limit: int | None = None
    default_log_level: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("default_presets", "default_seeds", "expected_root_causes", "expected_suspicious_backends"):
            data[key] = list(data[key])
        return data

    def experiment_defaults(self) -> dict[str, str | int]:
        return {
            "target_suite": self.target_suite,
            "presets": ",".join(self.default_presets),
            "cases": self.default_cases,
            "seeds": ",".join(str(seed) for seed in self.default_seeds),
            "evidence_mode": "historical",
            "known_bug_id": self.bug_id,
            "target_version": self.target_version,
        }


HISTORICAL_BUGS: tuple[HistoricalBugSpec, ...] = (
    HistoricalBugSpec(
        bug_id="duckdb-3015",
        project="duckdb/duckdb",
        status="candidate",
        target_suite="cross_family",
        target_version="duckdb==0.3.1",
        fixed_version="duckdb==0.3.2",
        issue_url="https://github.com/duckdb/duckdb/issues/3015",
        default_presets=("fixture_replay",),
        default_cases=1,
        default_seeds=(3015,),
        expected_root_causes=("ordering_or_limit",),
        expected_suspicious_backends=("duckdb",),
        notes=(
            "Official fixture replay using the generic fixture import path and per-column "
            "sort/null-order DSL. Keep as a candidate until the vulnerable DuckDB 0.3.1 "
            "Python package can be replayed by the final Python>=3.10 code path without "
            "a compatibility shim; a Python 3.10 isolated source build failed locally."
        ),
        replay_kind="fixture",
        fixture_spec="experiments/historical_replays/duckdb-3015.fixture.json",
        fixture_env="DATADIFF_DUCKDB_3015_FIXTURE",
    ),
    HistoricalBugSpec(
        bug_id="duckdb-22075",
        project="duckdb/duckdb",
        status="confirmed_fixed",
        target_suite="cross_family",
        target_version="duckdb==1.5.2",
        fixed_version="duckdb==1.5.3.dev26; upstream merge commit 5909259229afa28a33cee6075dc37fa602477325",
        issue_url="https://github.com/duckdb/duckdb/issues/22075",
        default_presets=("join_groupby_stress",),
        default_cases=20,
        default_seeds=(22075,),
        expected_root_causes=("groupby_aggregation",),
        expected_suspicious_backends=("duckdb",),
        notes=(
            "General join/groupby/join/groupby/global-aggregate stress replay. "
            "The bug is nondeterministic on DuckDB 1.5.2, so the default replay repeats "
            "the same generated stress shape for multiple cases. A 20-case local replay "
            "with DuckDB 1.5.2 produced groupby_aggregation@duckdb; the same 20-case "
            "replay with DuckDB 1.5.3.dev26 produced no findings."
        ),
    ),
    HistoricalBugSpec(
        bug_id="duckdb-22656",
        project="duckdb/duckdb",
        status="confirmed_fixed",
        target_suite="duckdb_storage_cross",
        target_version="duckdb==1.5.2",
        fixed_version="duckdb==1.5.3; upstream merge commit 67af30b260d7f0d8ad33092fc8e45a2ee87946c0",
        issue_url="https://github.com/duckdb/duckdb/issues/22656",
        default_presets=("storage_offset",),
        default_cases=2,
        default_seeds=(22656,),
        expected_root_causes=("ordering_or_limit",),
        expected_suspicious_backends=("duckdb_persistent",),
        notes=(
            "General storage-backed ORDER BY/OFFSET replay. The profile emits a persisted "
            "single-column table with 300000 rows and deterministic OFFSET variants over "
            "the duckdb_persistent target. A vulnerable DuckDB 1.5.2 replay is expected "
            "to produce ordering_or_limit@duckdb_persistent; DuckDB 1.5.3 is the locally "
            "verified fixed package for the same final code path."
        ),
        default_artifact_limit=0,
        default_log_level="minimal",
    ),
    HistoricalBugSpec(
        bug_id="datafusion-22190",
        project="apache/datafusion",
        status="pending_merge",
        target_suite="datafusion_cross",
        target_version="pre-fix-version-required",
        fixed_version="pending-upstream-merge",
        issue_url="https://github.com/apache/datafusion/issues/22190",
        default_presets=(
            "baseline",
            "discovery_guided",
            "filter_null_agg_topk",
            "join_null_agg_topk",
            "join_null_key_topk",
            "join_filter_groupby",
            "join_null_sort",
        ),
        default_cases=1000,
        default_seeds=(1, 1001, 2001),
        expected_root_causes=("grouped_topk_null_sort_key",),
        expected_suspicious_backends=("datafusion",),
        notes=(
            "Tracked as pending until the upstream fix is merged/released and a vulnerable "
            "target version is recorded. Do not count as confirmed historical replay evidence yet."
        ),
    ),
    HistoricalBugSpec(
        bug_id="duckdb-11261",
        project="duckdb/duckdb",
        status="candidate",
        target_suite="duckdb_storage_cross",
        target_version="duckdb-pre-fix-resource-boundary",
        fixed_version="",
        issue_url="https://github.com/duckdb/duckdb/issues/11261",
        default_presets=("wide_offset_topk",),
        default_cases=10,
        default_seeds=(11261,),
        expected_root_causes=("ordering_or_limit",),
        expected_suspicious_backends=("duckdb_persistent",),
        notes=(
            "Resource-sensitive ORDER BY/LIMIT/OFFSET over wide tables. The default "
            "profile keeps row counts modest for CI and can be scaled manually for "
            "performance replay; do not count as confirmed historical evidence."
        ),
        default_artifact_limit=0,
        default_log_level="minimal",
    ),
    HistoricalBugSpec(
        bug_id="arrow-42231",
        project="apache/arrow",
        status="candidate",
        target_suite="arrow_cross",
        target_version="pyarrow-pre-fix-large-groupby",
        fixed_version="",
        issue_url="https://github.com/apache/arrow/issues/42231",
        default_presets=("join_groupby_stress",),
        default_cases=5,
        default_seeds=(42231,),
        expected_root_causes=("groupby_aggregation",),
        expected_suspicious_backends=("pyarrow",),
        notes=(
            "Targets PyArrow Table.group_by/aggregate key-collision behavior on larger "
            "inputs using the existing join/groupby stress generator. Keep as a "
            "candidate study until a vulnerable PyArrow version is pinned locally."
        ),
        default_artifact_limit=5,
        default_log_level="minimal",
    ),
    HistoricalBugSpec(
        bug_id="datafusion-22441",
        project="apache/datafusion",
        status="pending_merge",
        target_suite="datafusion_cross",
        target_version="datafusion-cli-v53.1.0",
        fixed_version="pending-upstream-fix",
        issue_url="https://github.com/apache/datafusion/issues/22441",
        default_presets=(
            "join_null_truth_filter",
            "discovery_guided",
        ),
        default_cases=500,
        default_seeds=(1, 1001, 2001),
        expected_root_causes=("outer_join_truth_filter",),
        expected_suspicious_backends=("datafusion",),
        notes=(
            "Targets LEFT JOIN rows preserved by NOT ((right_col > literal) IS TRUE). "
            "Do not count as confirmed historical replay evidence until an upstream fix "
            "or documented expected behavior is available."
        ),
    ),
)


def list_historical_bugs(*, include_pending: bool = False) -> list[HistoricalBugSpec]:
    return [
        spec
        for spec in HISTORICAL_BUGS
        if include_pending or spec.status == "confirmed_fixed"
    ]


def get_historical_bug(bug_id: str, *, include_pending: bool = False) -> HistoricalBugSpec | None:
    for spec in list_historical_bugs(include_pending=include_pending):
        if spec.bug_id == bug_id:
            return spec
    return None
