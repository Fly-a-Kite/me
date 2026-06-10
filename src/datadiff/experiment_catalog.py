from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.preset_catalog import build_catalog_preset
from datadiff.preset_catalog import build_experiment_config
from datadiff.preset_catalog import catalog_preset_semantic_focus


TARGET_SUITE_SCOPE_KIND: dict[str, str] = {
    "arrow_cross": "cross_ecosystem",
    "core": "core",
    "core_arrow": "core",
    "core_datafusion": "core",
    "core_lazy": "core",
    "cross_family": "cross_ecosystem",
    "dataframe_lazy": "dataframe_consistency",
    "datafusion_cross": "cross_ecosystem",
    "duckdb_storage_cross": "sql_oriented",
    "embedded_sql": "sql_oriented",
    "embedded_sql_cross": "sql_oriented",
    "latest_all_engines": "cross_ecosystem",
    "latest_no_datafusion": "cross_ecosystem",
    "polars_cross": "dataframe_consistency",
    "seeded_filter": "seeded_fault_injection",
    "seeded_groupby": "seeded_fault_injection",
    "seeded_join": "seeded_fault_injection",
    "seeded_mutate": "seeded_fault_injection",
}

FINAL_PROTOCOL_TRACKS: tuple[str, ...] = (
    "validation",
    "live",
    "historical",
    "seeded",
    "ablation",
    "comparison",
)

EXPERIMENT_EVIDENCE_MODES: tuple[str, ...] = FINAL_PROTOCOL_TRACKS
FIXTURE_REPLAY_EVIDENCE_MODES: tuple[str, ...] = (
    "live",
    "historical",
    "seeded",
    "validation",
)

EVIDENCE_MODE_COUNTING_POLICY: dict[str, str] = {
    "validation": "Use only as pre-freeze harness smoke; do not count as final bug evidence.",
    "historical": "Count only promoted confirmed_fixed historical specs; pending case studies are not counted.",
    "seeded": "Use for sensitivity only; never count seeded faults as real backend bugs.",
    "ablation": "Use for module ablation RQ tables only; do not count candidates as live bug evidence.",
    "comparison": "Use for baseline/related-scope comparison RQ tables only; do not count candidates as live bug evidence.",
    "live": "Count candidate families separately from maintainer-confirmed or fixed bugs.",
}

EVIDENCE_MODE_REPLAY_DEFAULT: dict[str, bool] = {
    "validation": False,
    "live": False,
    "historical": True,
    "seeded": False,
    "ablation": False,
    "comparison": False,
}


def scope_kind_for_suite(suite: str) -> str:
    return TARGET_SUITE_SCOPE_KIND.get(suite, "")


def counting_policy_for_evidence_mode(evidence_mode: str) -> str:
    return EVIDENCE_MODE_COUNTING_POLICY.get(
        str(evidence_mode or "").strip(),
        EVIDENCE_MODE_COUNTING_POLICY["live"],
    )


def replay_bug_enabled_by_default(evidence_mode: str) -> bool:
    return EVIDENCE_MODE_REPLAY_DEFAULT.get(str(evidence_mode or "").strip(), False)


def _suite_scope_map(suites: tuple[str, ...]) -> dict[str, str]:
    return {
        suite: scope_kind
        for suite in suites
        if (scope_kind := scope_kind_for_suite(suite))
    }


def _string_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _uniform_scope_kind(
    selected_suites: tuple[str, ...],
    *,
    scope_by_target_suite: dict[str, str],
) -> str:
    scope_kinds = {
        str(scope_by_target_suite.get(suite, "") or "").strip()
        for suite in selected_suites
        if str(scope_by_target_suite.get(suite, "") or "").strip()
    }
    if len(scope_kinds) == 1:
        return next(iter(scope_kinds))
    return ""


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    id: str
    preset: str
    title: str = ""
    base_preset: str = ""
    comparison_role: str = ""
    scope_kind: str = ""
    component_focus: str = ""
    overlays: tuple[str, ...] = ()
    semantic_focus_families: tuple[str, ...] = ()
    semantic_focus_signals: tuple[str, ...] = ()
    factors: dict[str, Any] = field(default_factory=dict)
    oracle_profile: str = ""
    rq_tags: tuple[str, ...] = ()
    analysis_tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.semantic_focus_families or self.semantic_focus_signals:
            return
        preset_families, preset_signals = catalog_preset_semantic_focus(self.preset)
        if preset_families:
            object.__setattr__(self, "semantic_focus_families", preset_families)
        if preset_signals:
            object.__setattr__(self, "semantic_focus_signals", preset_signals)

    def to_meta(self) -> dict[str, Any]:
        return {
            "variant_id": self.id,
            "variant_title": self.title or self.id,
            "preset": self.preset,
            "base_preset": self.base_preset or self.preset,
            "comparison_role": self.comparison_role,
            "scope_kind": self.scope_kind,
            "component_focus": self.component_focus,
            "overlays": list(self.overlays),
            "semantic_focus_families": list(self.semantic_focus_families),
            "semantic_focus_signals": list(self.semantic_focus_signals),
            "factors": dict(self.factors),
            "oracle_profile": self.oracle_profile,
            "rq_tags": list(self.rq_tags),
            "analysis_tags": list(self.analysis_tags),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ExperimentCampaign:
    suite: str
    preset: str
    purpose: str
    counts_as_real_bugs: bool | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    id: str
    title: str
    track: str
    purpose: str
    evidence_mode: str
    target_suites: tuple[str, ...]
    variants: tuple[ExperimentVariant, ...]
    comparison_group: str = ""
    rq_tags: tuple[str, ...] = ()
    analysis_tags: tuple[str, ...] = ()
    counts_as_real_bugs: bool = False
    notes: str = ""
    scope_by_target_suite: dict[str, str] = field(default_factory=dict)
    campaigns: tuple[ExperimentCampaign, ...] = ()

    def __post_init__(self) -> None:
        preset_names = {variant.preset for variant in self.variants}
        campaign_keys: set[tuple[str, str]] = set()
        for campaign in self.campaigns:
            if campaign.suite not in self.target_suites:
                raise ValueError(
                    f"campaign suite {campaign.suite!r} is not registered in matrix {self.id!r}"
                )
            if campaign.preset not in preset_names:
                raise ValueError(
                    f"campaign preset {campaign.preset!r} is not registered in matrix {self.id!r}"
                )
            key = (campaign.suite, campaign.preset)
            if key in campaign_keys:
                raise ValueError(f"duplicate campaign {key!r} in matrix {self.id!r}")
            campaign_keys.add(key)

    def to_experiment_meta(self, *, target_suites: tuple[str, ...] | None = None) -> dict[str, Any]:
        selected_suites = tuple(target_suites or self.target_suites)
        scope_by_target_suite = {
            suite: scope_kind
            for suite in selected_suites
            if (scope_kind := self.scope_by_target_suite.get(suite, ""))
        }
        uniform_scope_kind = _uniform_scope_kind(
            selected_suites,
            scope_by_target_suite=self.scope_by_target_suite,
        )
        return {
            "track": self.track,
            "matrix_id": self.id,
            "matrix_title": self.title,
            "purpose": self.purpose,
            "comparison_group": self.comparison_group or self.id,
            "rq_tags": list(self.rq_tags),
            "analysis_tags": list(self.analysis_tags),
            "counts_as_real_bugs": self.counts_as_real_bugs,
            "notes": self.notes,
            "target_suites": list(selected_suites),
            "scope_kind": uniform_scope_kind,
            "scope_by_target_suite": scope_by_target_suite,
            "variant_by_preset": {
                variant.preset: {
                    **variant.to_meta(),
                    "scope_kind": variant.scope_kind or uniform_scope_kind,
                }
                for variant in self.variants
            },
        }

    def command_experiment_meta(
        self,
        *,
        target_suites: tuple[str, ...] | None = None,
        preset: str | None = None,
        counts_as_real_bugs: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = self.to_experiment_meta(target_suites=target_suites)
        out: dict[str, Any] = {
            "track": meta["track"],
            "matrix_id": meta["matrix_id"],
            "matrix_title": meta["matrix_title"],
            "purpose": meta["purpose"],
            "comparison_group": meta["comparison_group"],
            "rq_tags": list(meta["rq_tags"]),
            "analysis_tags": list(meta["analysis_tags"]),
            "counts_as_real_bugs": (
                meta["counts_as_real_bugs"] if counts_as_real_bugs is None else bool(counts_as_real_bugs)
            ),
            "target_suites": list(target_suites or self.target_suites),
            "scope_by_target_suite": dict(meta["scope_by_target_suite"]),
        }
        if preset is not None:
            variant = dict(meta["variant_by_preset"][preset])
            variant.pop("preset", None)
            out["variant"] = variant
        if extra:
            out.update(extra)
        return out

    def command_experiment_meta_for_campaign(self, campaign: ExperimentCampaign) -> dict[str, Any]:
        variant = next((item for item in self.variants if item.preset == campaign.preset), None)
        scope_kind = str(getattr(variant, "scope_kind", "") or "").strip() or scope_kind_for_suite(campaign.suite)
        return self.command_experiment_meta(
            target_suites=(campaign.suite,),
            preset=campaign.preset,
            counts_as_real_bugs=(
                self.counts_as_real_bugs
                if campaign.counts_as_real_bugs is None
                else campaign.counts_as_real_bugs
            ),
            extra={
                "scope_kind": scope_kind,
                "variant_scope_kind": scope_kind,
            },
        )


def build_historical_experiment_meta(spec: Any) -> dict[str, Any]:
    bug_id = str(getattr(spec, "bug_id", "") or "")
    status = str(getattr(spec, "status", "") or "")
    replay_kind = str(getattr(spec, "replay_kind", "experiment") or "experiment")
    suite = str(getattr(spec, "target_suite", "") or "")
    scope_kind = scope_kind_for_suite(suite)
    default_presets = _string_tuple(getattr(spec, "default_presets", ()))
    base_preset = default_presets[0] if default_presets else ("fixture_replay" if replay_kind == "fixture" else "")
    expected_root_causes = _string_tuple(getattr(spec, "expected_root_causes", ()))
    expected_suspicious_backends = _string_tuple(getattr(spec, "expected_suspicious_backends", ()))
    analysis_tags = tuple(
        tag
        for tag in ("historical", replay_kind, status)
        if tag
    )
    return _historical_experiment_meta(
        bug_id=bug_id,
        suite=suite,
        scope_kind=scope_kind,
        status=status,
        replay_kind=replay_kind,
        target_version=str(getattr(spec, "target_version", "") or ""),
        fixed_version=str(getattr(spec, "fixed_version", "") or ""),
        issue_url=str(getattr(spec, "issue_url", "") or ""),
        project=str(getattr(spec, "project", "") or ""),
        base_preset=base_preset,
        analysis_tags=analysis_tags,
        expected_root_causes=expected_root_causes,
        expected_suspicious_backends=expected_suspicious_backends,
        notes=str(getattr(spec, "notes", "") or ""),
        counts_as_real_bugs=status == "confirmed_fixed",
    )


def _historical_variant_meta(
    *,
    bug_id: str,
    base_preset: str,
    scope_kind: str,
    status: str,
    replay_kind: str,
    expected_root_causes: tuple[str, ...],
    expected_suspicious_backends: tuple[str, ...],
    analysis_tags: tuple[str, ...],
    notes: str,
) -> dict[str, Any]:
    return {
        "variant_id": bug_id,
        "variant_title": bug_id or "historical_replay",
        "base_preset": base_preset,
        "comparison_role": "contrast",
        "component_focus": "",
        "overlays": ["enable_replay_bug"],
        "factors": {
            "historical_bug_id": bug_id,
            "historical_status": status,
            "replay_kind": replay_kind,
            "expected_root_causes": list(expected_root_causes),
            "expected_suspicious_backends": list(expected_suspicious_backends),
        },
        "oracle_profile": "differential",
        "rq_tags": [],
        "analysis_tags": list(analysis_tags),
        "notes": notes,
        "scope_kind": scope_kind,
    }


def _historical_experiment_meta(
    *,
    bug_id: str,
    suite: str,
    scope_kind: str,
    status: str,
    replay_kind: str,
    target_version: str,
    fixed_version: str,
    issue_url: str,
    project: str,
    base_preset: str,
    analysis_tags: tuple[str, ...],
    expected_root_causes: tuple[str, ...],
    expected_suspicious_backends: tuple[str, ...],
    notes: str,
    counts_as_real_bugs: bool,
) -> dict[str, Any]:
    meta = {
        "track": "historical",
        "matrix_id": "historical_replay",
        "matrix_title": "Historical Replay",
        "purpose": (
            "Replay previously reported bugs on vulnerable target versions using the same final "
            "generator, runner, normalizer, oracle, and reporting pipeline."
        ),
        "comparison_group": "historical_replay",
        "rq_tags": [],
        "analysis_tags": list(analysis_tags),
        "counts_as_real_bugs": counts_as_real_bugs,
        "target_suites": [suite] if suite else [],
        "scope_by_target_suite": {suite: scope_kind} if suite and scope_kind else {},
        "variant": _historical_variant_meta(
            bug_id=bug_id,
            base_preset=base_preset,
            scope_kind=scope_kind,
            status=status,
            replay_kind=replay_kind,
            expected_root_causes=expected_root_causes,
            expected_suspicious_backends=expected_suspicious_backends,
            analysis_tags=analysis_tags,
            notes=notes,
        ),
        "historical": {
            "bug_id": bug_id,
            "project": project,
            "status": status,
            "replay_kind": replay_kind,
            "target_version": target_version,
            "fixed_version": fixed_version,
            "issue_url": issue_url,
            "expected_root_causes": list(expected_root_causes),
            "expected_suspicious_backends": list(expected_suspicious_backends),
        },
    }
    if bug_id:
        meta["known_bug_id"] = bug_id
    if suite and scope_kind:
        meta["scope_kind"] = scope_kind
    if target_version:
        meta["target_version"] = target_version
    return meta


def resolve_historical_experiment_meta(
    *,
    known_bug_id: str,
    target_suite: str,
    target_version: str,
    include_pending: bool = True,
) -> dict[str, Any]:
    bug_id = str(known_bug_id or "").strip()
    if not bug_id:
        return {}
    try:
        from datadiff.historical import get_historical_bug

        spec = get_historical_bug(bug_id, include_pending=include_pending)
    except Exception:
        spec = None
    if spec is not None:
        meta = build_historical_experiment_meta(spec)
    else:
        scope_kind = scope_kind_for_suite(str(target_suite or ""))
        meta = _historical_experiment_meta(
            bug_id=bug_id,
            suite=target_suite,
            scope_kind=scope_kind,
            status="unregistered",
            replay_kind="experiment",
            target_version=target_version,
            fixed_version="",
            issue_url="",
            project="",
            base_preset="",
            analysis_tags=("historical",),
            expected_root_causes=(),
            expected_suspicious_backends=(),
            notes="",
            counts_as_real_bugs=False,
        )
    historical = meta.get("historical", {})
    if isinstance(historical, dict):
        historical = dict(historical)
        if target_version and not str(historical.get("target_version", "") or "").strip():
            historical["target_version"] = target_version
        meta["historical"] = historical
    return meta


def registered_experiment_meta_defaults(
    *,
    evidence_mode: str,
    known_bug_id: str = "",
    target_suite: str = "",
    target_version: str = "",
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    return registered_experiment_meta_for_run(
        evidence_mode=evidence_mode,
        known_bug_id=known_bug_id,
        target_suite=target_suite,
        target_version=target_version,
        include_pending_historical=include_pending_historical,
    )


def resolve_experiment_meta_defaults(
    *,
    evidence_mode: str,
    known_bug_id: str = "",
    target_suite: str = "",
    target_version: str = "",
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    return registered_experiment_meta_defaults(
        evidence_mode=evidence_mode,
        known_bug_id=known_bug_id,
        target_suite=target_suite,
        target_version=target_version,
        include_pending_historical=include_pending_historical,
    )


FINAL_VALIDATION_MATRIX = ExperimentMatrix(
    id="final_validation",
    title="Final Validation Smoke",
    track="validation",
    purpose=(
        "Short pre-freeze validation over live target families to catch adapter, oracle, "
        "classification, and evidence-pipeline noise before 24h runs."
    ),
    evidence_mode="validation",
    target_suites=(
        "datafusion_cross",
        "polars_cross",
        "arrow_cross",
        "embedded_sql_cross",
        "latest_no_datafusion",
    ),
    variants=(
        ExperimentVariant(
            id="baseline",
            preset="baseline",
            comparison_role="baseline",
            oracle_profile="differential",
        ),
        ExperimentVariant(
            id="workflow",
            preset="workflow",
            oracle_profile="differential",
            factors={"generator_profile": "workflow"},
            analysis_tags=("workflow",),
        ),
        ExperimentVariant(
            id="live_common_api_workflow_metamorphic",
            preset="live_common_api_workflow_metamorphic",
            base_preset="live_common_api_workflow",
            overlays=("enable_metamorphic_oracle", "candidate_pool_8", "metamorphic_variant_limit_6"),
            factors={"generator_profile": "common_api_workflow", "guidance": "guided", "metamorphic_oracle": True},
            oracle_profile="both",
            analysis_tags=("workflow", "metamorphic", "validation"),
        ),
        ExperimentVariant(
            id="live_deep_organic_metamorphic",
            preset="live_deep_organic_metamorphic",
            base_preset="live_deep_organic",
            overlays=("enable_metamorphic_oracle", "candidate_pool_10", "metamorphic_variant_limit_8"),
            factors={"generator_profile": "discovery", "guidance": "guided", "metamorphic_oracle": True},
            oracle_profile="both",
            analysis_tags=("guided", "metamorphic", "validation"),
        ),
    ),
    comparison_group="validation_smoke",
    analysis_tags=("validation", "readiness_gate"),
    counts_as_real_bugs=False,
    notes="Validation evidence is a readiness gate only and must not be counted as final bug evidence.",
    scope_by_target_suite=_suite_scope_map(
        (
            "datafusion_cross",
            "polars_cross",
            "arrow_cross",
            "embedded_sql_cross",
            "latest_no_datafusion",
        )
    ),
)


FINAL_LIVE_DISCOVERY_MATRIX = ExperimentMatrix(
    id="live_discovery",
    title="Latest-Version Live Discovery",
    track="live",
    purpose="Latest-version fresh discovery over live backend families under the frozen final harness.",
    evidence_mode="live",
    target_suites=(
        "datafusion_cross",
        "dataframe_lazy",
        "arrow_cross",
        "embedded_sql",
        "latest_all_engines",
        "latest_no_datafusion",
        "polars_cross",
        "embedded_sql_cross",
    ),
    variants=(
        ExperimentVariant(
            id="live_datafusion",
            preset="live_datafusion",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "guided"),
        ),
        ExperimentVariant(
            id="live_datafusion_fresh",
            preset="live_datafusion_fresh",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "guided", "fresh_safe"),
        ),
        ExperimentVariant(
            id="live_polars_lazy",
            preset="live_polars_lazy",
            scope_kind="dataframe_consistency",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "lazy_consistency"),
        ),
        ExperimentVariant(
            id="live_arrow",
            preset="live_arrow",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "guided"),
        ),
        ExperimentVariant(
            id="live_embedded_sql",
            preset="live_embedded_sql",
            scope_kind="sql_oriented",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "sql_scope"),
        ),
        ExperimentVariant(
            id="live_cross_family",
            preset="live_cross_family",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1", "RQ6"),
            analysis_tags=("live", "cross_ecosystem"),
        ),
        ExperimentVariant(
            id="live_polars_issue_focus",
            preset="live_polars_issue_focus",
            scope_kind="dataframe_consistency",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "issue_focus"),
        ),
        ExperimentVariant(
            id="live_duckdb_issue_focus",
            preset="live_duckdb_issue_focus",
            scope_kind="sql_oriented",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "issue_focus", "sql_scope"),
        ),
        ExperimentVariant(
            id="live_arrow_issue_focus",
            preset="live_arrow_issue_focus",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "issue_focus"),
        ),
        ExperimentVariant(
            id="live_issue_focus",
            preset="live_issue_focus",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1",),
            analysis_tags=("live", "issue_focus", "cross_ecosystem"),
        ),
    ),
    comparison_group="latest_live_discovery",
    rq_tags=("RQ1",),
    analysis_tags=("live", "latest_version"),
    counts_as_real_bugs=True,
    notes="Live latest-version findings count as candidate/confirmed backend bug evidence after family deduplication.",
    scope_by_target_suite=_suite_scope_map(
        (
            "datafusion_cross",
            "dataframe_lazy",
            "arrow_cross",
            "embedded_sql",
            "latest_all_engines",
            "latest_no_datafusion",
            "polars_cross",
            "embedded_sql_cross",
        )
    ),
    campaigns=(
        ExperimentCampaign(
            suite="datafusion_cross",
            preset="live_datafusion",
            purpose="Latest-version DataFusion differential discovery against pandas and DuckDB references.",
        ),
        ExperimentCampaign(
            suite="datafusion_cross",
            preset="live_datafusion_fresh",
            purpose="Latest-version DataFusion fresh discovery over non-groupby surfaces using the same shared harness.",
        ),
        ExperimentCampaign(
            suite="dataframe_lazy",
            preset="live_polars_lazy",
            purpose="Latest-version Polars eager/lazy consistency discovery.",
        ),
        ExperimentCampaign(
            suite="arrow_cross",
            preset="live_arrow",
            purpose="Latest-version Arrow/PyArrow cross-family discovery.",
        ),
        ExperimentCampaign(
            suite="embedded_sql",
            preset="live_embedded_sql",
            purpose="Latest-version embedded SQL cross-engine discovery.",
        ),
        ExperimentCampaign(
            suite="latest_all_engines",
            preset="live_cross_family",
            purpose="Broad latest-version cross-family discovery over every implemented real backend.",
        ),
        ExperimentCampaign(
            suite="latest_no_datafusion",
            preset="live_cross_family",
            purpose="Broad latest-version cross-family discovery excluding DataFusion to avoid known DataFusion saturation.",
        ),
        ExperimentCampaign(
            suite="polars_cross",
            preset="live_polars_issue_focus",
            purpose="Latest-version Polars-focused discovery using fresh-safe issue-inspired semantic sketches.",
        ),
        ExperimentCampaign(
            suite="embedded_sql_cross",
            preset="live_duckdb_issue_focus",
            purpose="Latest-version DuckDB/SQLite/Pandas discovery using fresh-safe issue-inspired SQL sketches.",
        ),
        ExperimentCampaign(
            suite="arrow_cross",
            preset="live_arrow_issue_focus",
            purpose="Latest-version Arrow/PyArrow discovery using fresh-safe issue-inspired Arrow sketches.",
        ),
        ExperimentCampaign(
            suite="latest_no_datafusion",
            preset="live_issue_focus",
            purpose="Broad non-DataFusion latest-version discovery over fresh-safe issue-inspired sketches.",
        ),
    ),
)


FINAL_SEEDED_SENSITIVITY_MATRIX = ExperimentMatrix(
    id="seeded_sensitivity",
    title="Seeded Sensitivity",
    track="seeded",
    purpose="Measure injected-fault sensitivity and time-to-first under controlled seeded regressions.",
    evidence_mode="seeded",
    target_suites=("seeded_filter", "seeded_groupby", "seeded_join", "seeded_mutate"),
    variants=(
        ExperimentVariant(
            id="baseline",
            preset="baseline",
            comparison_role="baseline",
            oracle_profile="differential",
            analysis_tags=("seeded", "baseline"),
        ),
        ExperimentVariant(
            id="guided_filter",
            preset="guided_filter",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("enable_guidance", "target_filter"),
            factors={"guidance": "guided", "guidance_targets": ["filter"]},
            oracle_profile="differential",
            analysis_tags=("seeded", "guided"),
        ),
        ExperimentVariant(
            id="guided_groupby",
            preset="guided_groupby",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("enable_guidance", "target_groupby"),
            factors={"guidance": "guided", "guidance_targets": ["groupby", "aggregation"]},
            oracle_profile="differential",
            analysis_tags=("seeded", "guided"),
        ),
        ExperimentVariant(
            id="guided_join",
            preset="guided_join",
            base_preset="discovery_no_groupby",
            comparison_role="contrast",
            overlays=("enable_guidance", "target_join"),
            factors={
                "generator_profile": "discovery_no_groupby",
                "guidance": "guided",
                "guidance_targets": ["join", "sort_limit"],
            },
            oracle_profile="differential",
            analysis_tags=("seeded", "guided"),
        ),
        ExperimentVariant(
            id="guided_mutate",
            preset="guided_mutate",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("enable_guidance", "target_mutate"),
            factors={"guidance": "guided", "guidance_targets": ["mutate", "expressions"]},
            oracle_profile="differential",
            analysis_tags=("seeded", "guided"),
        ),
    ),
    comparison_group="seeded_sensitivity",
    analysis_tags=("seeded", "method_validation"),
    counts_as_real_bugs=False,
    notes="Seeded runs are method-validation evidence only and must not be counted as real backend bugs.",
    scope_by_target_suite=_suite_scope_map(("seeded_filter", "seeded_groupby", "seeded_join", "seeded_mutate")),
)


FINAL_MODULE_ABLATION_MATRIX = ExperimentMatrix(
    id="module_ablation",
    title="Module Ablation",
    track="ablation",
    purpose=(
        "Quantify sensitivity of type-aware generation, semantic normalization, feedback, "
        "reducer, and oracle composition across core target families."
    ),
    evidence_mode="ablation",
    target_suites=("core", "core_lazy", "core_datafusion", "core_arrow", "latest_no_datafusion"),
    variants=(
        ExperimentVariant(
            id="baseline",
            preset="baseline",
            comparison_role="baseline",
            oracle_profile="differential",
            rq_tags=("RQ2", "RQ3", "RQ4", "RQ5"),
            analysis_tags=("ablation", "baseline"),
        ),
        ExperimentVariant(
            id="no_type_aware",
            preset="no_type_aware",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="type_aware_generation",
            overlays=("disable_type_aware_generation",),
            factors={"type_aware_generation": False},
            oracle_profile="differential",
            rq_tags=("RQ2", "RQ4"),
            analysis_tags=("ablation", "noise_control"),
        ),
        ExperimentVariant(
            id="no_normalizer",
            preset="no_normalizer",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="semantic_normalizer",
            overlays=("disable_normalizer",),
            factors={"semantic_normalizer": False},
            oracle_profile="differential",
            rq_tags=("RQ2",),
            analysis_tags=("ablation", "noise_control"),
        ),
        ExperimentVariant(
            id="no_feedback",
            preset="no_feedback",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="feedback_corpus",
            overlays=("disable_feedback_corpus",),
            factors={"feedback_corpus": False},
            oracle_profile="differential",
            rq_tags=("RQ4",),
            analysis_tags=("ablation", "efficiency"),
        ),
        ExperimentVariant(
            id="metamorphic",
            preset="metamorphic",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="metamorphic_oracle",
            overlays=("enable_metamorphic_oracle",),
            factors={"metamorphic_oracle": True, "oracle_mode": "both"},
            oracle_profile="both",
            rq_tags=("RQ3", "RQ4"),
            analysis_tags=("ablation", "oracle_complementarity"),
        ),
        ExperimentVariant(
            id="oracle_only_metamorphic",
            preset="oracle_only_metamorphic",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="differential_oracle",
            overlays=("disable_differential_oracle", "enable_metamorphic_oracle"),
            factors={"differential_oracle": False, "metamorphic_oracle": True, "oracle_mode": "metamorphic"},
            oracle_profile="metamorphic_only",
            rq_tags=("RQ3",),
            analysis_tags=("ablation", "oracle_complementarity"),
        ),
        ExperimentVariant(
            id="reducer",
            preset="reducer",
            base_preset="baseline",
            comparison_role="contrast",
            component_focus="reducer",
            overlays=("enable_reducer",),
            factors={"reducer": True},
            oracle_profile="differential",
            rq_tags=("RQ5",),
            analysis_tags=("ablation", "actionability"),
        ),
    ),
    comparison_group="module_ablation",
    rq_tags=("RQ2", "RQ3", "RQ4", "RQ5"),
    analysis_tags=("ablation",),
    counts_as_real_bugs=False,
    notes="Ablation findings require the same live confirmation pipeline before they can be counted as real bugs.",
    scope_by_target_suite=_suite_scope_map(("core", "core_lazy", "core_datafusion", "core_arrow", "latest_no_datafusion")),
)


FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX = ExperimentMatrix(
    id="adaptive_component_ablation",
    title="Adaptive Component Ablation",
    track="ablation",
    purpose=(
        "Quantify closed-loop adaptive components while keeping the generator profile, "
        "target suite, oracle mode, and seed budget fixed."
    ),
    evidence_mode="ablation",
    target_suites=("datafusion_cross",),
    variants=(
        ExperimentVariant(
            id="adaptive_reference",
            preset="live_deep_organic",
            comparison_role="baseline",
            component_focus="adaptive_closed_loop",
            factors={
                "adaptive_closed_loop": True,
                "scheduler_learning": True,
                "online_reward_model": True,
                "continual_learning": True,
                "active_learning": True,
                "ir_rewrite_mutations": True,
                "operator_swarm": True,
                "divergence_conditioned": True,
                "shrink_mutations": True,
                "value_catalog": True,
                "quality_archive": True,
                "hierarchical_archive": True,
                "bd_axis_bandit": True,
                "bayesian_exploration": True,
                "seed_quota": True,
                "seed_energy_batch": True,
                "seed_energy_tier": True,
                "per_operator_energy": True,
                "lineage_rarity": True,
                "minhash_dedup": True,
                "disagreement_bd_axis": True,
                "lhs_seeding": True,
                "champion_corpus": True,
                "champion_graft_donor": True,
                "backend_pair_learning": True,
                "runtime_cost_learning": True,
                "cost_normalized_reward": True,
                "scheduler_annealing": True,
            },
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "reference"),
            notes="Reference adaptive run with all adaptive components enabled.",
        ),
        ExperimentVariant(
            id="no_scheduler_learning",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="scheduler_learning",
            factors={"scheduler_learning": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "contextual_bandit"),
            notes=(
                "Disable the cross-batch contextual scheduler while preserving the same "
                "generator preset, target suite, oracle, and seed budget."
            ),
        ),
        ExperimentVariant(
            id="no_scheduler_annealing",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="scheduler_annealing",
            factors={"scheduler_annealing": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "annealing"),
            notes=(
                "Disable annealed scheduler selection while keeping contextual bandit, "
                "online reward, continual learning, and quality-diversity enabled."
            ),
        ),
        ExperimentVariant(
            id="no_online_reward_model",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="online_reward_model",
            factors={"online_reward_model": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "online_reward_model"),
            notes=(
                "Disable online reward-model prediction and weight updates while keeping "
                "basic contextual-bandit arm statistics enabled."
            ),
        ),
        ExperimentVariant(
            id="no_continual_learning",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="continual_learning",
            factors={"continual_learning": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "continual_learning"),
            notes=(
                "Disable cross-version transfer and continual-priority signals while keeping "
                "the same live adaptive schedule and local learning feedback."
            ),
        ),
        ExperimentVariant(
            id="no_runtime_cost_learning",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="runtime_cost_learning",
            factors={"runtime_cost_learning": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "efficiency"),
            notes=(
                "Disable runtime-cost feedback while leaving scheduler learning enabled, "
                "isolating throughput/invalid-rate contribution from cost-aware learning."
            ),
        ),
        ExperimentVariant(
            id="no_quality_archive",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="quality_archive",
            factors={"quality_archive": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "quality_diversity"),
            notes=(
                "Disable the MAP-Elites quality-diversity archive while keeping the adaptive "
                "scheduler and runtime-cost learning enabled, isolating diversity preservation."
            ),
        ),
        ExperimentVariant(
            id="no_bd_axis_bandit",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="bd_axis_bandit",
            factors={"bd_axis_bandit": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "behavioral_descriptor"),
            notes=(
                "Disable learned behavioral-descriptor axis weights while keeping the "
                "quality-diversity archive, scheduler learning, and runtime-cost learning enabled."
            ),
        ),
        ExperimentVariant(
            id="no_bayesian_exploration",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="bayesian_exploration",
            factors={"bayesian_exploration": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "good_turing"),
            notes=(
                "Disable Good-Turing discovery-rate feedback into adaptive exploration "
                "while keeping scheduler learning and quality-diversity enabled."
            ),
        ),
        ExperimentVariant(
            id="no_seed_energy_batch",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="seed_energy_batch",
            factors={"seed_energy_batch": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "seed_power_scheduling"),
            notes=(
                "Disable AFL-FAST-style seed-energy batching while keeping seed quota, "
                "quality-diversity archive, and scheduler learning enabled."
            ),
        ),
        ExperimentVariant(
            id="no_per_operator_energy",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="per_operator_energy",
            factors={"per_operator_energy": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "operator_power_scheduling"),
            notes=(
                "Disable AFL-FAST-style per-operator candidate energy while keeping "
                "seed-energy batching, quality-diversity archive, and scheduler learning enabled."
            ),
        ),
        ExperimentVariant(
            id="no_seed_energy_tier",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="seed_energy_tier",
            factors={"seed_energy_tier": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "seed_power_scheduling"),
            notes=(
                "Disable the learned seed-energy tier bandit while keeping seed-energy "
                "batching, seed quota, and the rest of the adaptive scheduler enabled."
            ),
        ),
        ExperimentVariant(
            id="no_ir_rewrite_mutations",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="ir_rewrite_mutations",
            factors={"ir_rewrite_mutations": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "typed_ir_rewrite"),
            notes=(
                "Disable typed IR rewrite mutations while keeping operator learning, "
                "seed-energy batching, and quality-diversity archive enabled."
            ),
        ),
        ExperimentVariant(
            id="no_operator_swarm",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="operator_swarm",
            factors={"operator_swarm": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "mutation_operator_swarm"),
            notes=(
                "Disable MOPT-style mutation-operator swarm selection while keeping "
                "operator reward accounting and the rest of the adaptive loop enabled."
            ),
        ),
        ExperimentVariant(
            id="no_divergence_conditioned",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="divergence_conditioned",
            factors={"divergence_conditioned": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "divergence_conditioned_mutation"),
            notes=(
                "Disable divergence-conditioned mutation affinity while keeping the "
                "same mutation operator pool and feedback loop enabled."
            ),
        ),
        ExperimentVariant(
            id="no_shrink_mutations",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="shrink_mutations",
            factors={"shrink_mutations": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "grow_shrink_mutation"),
            notes=(
                "Disable shrink-side mutation operators while keeping growth, typed IR "
                "rewrites, and operator learning enabled."
            ),
        ),
        ExperimentVariant(
            id="no_value_catalog",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="value_catalog",
            factors={"value_catalog": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "value_catalog"),
            notes=(
                "Disable learned adversarial value-catalog sampling while keeping "
                "operator learning, seed scheduling, and quality-diversity enabled."
            ),
        ),
        ExperimentVariant(
            id="no_hierarchical_archive",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="hierarchical_archive",
            factors={"hierarchical_archive": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "hierarchical_map_elites"),
            notes=(
                "Disable hierarchical MAP-Elites cell split/merge while preserving the "
                "base quality-diversity archive and adaptive scheduler."
            ),
        ),
        ExperimentVariant(
            id="no_seed_quota",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="seed_quota",
            factors={"seed_quota": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "seed_quota"),
            notes=(
                "Disable per-cluster seed-energy quotas while keeping AFL-FAST seed "
                "batching and quality-diversity archive feedback enabled."
            ),
        ),
        ExperimentVariant(
            id="no_lineage_rarity",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="lineage_rarity",
            factors={"lineage_rarity": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "lineage_scheduler"),
            notes=(
                "Disable lineage-rarity bonuses in seed selection while keeping the "
                "lineage graph and other feedback statistics enabled."
            ),
        ),
        ExperimentVariant(
            id="no_minhash_dedup",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="minhash_dedup",
            factors={"minhash_dedup": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "minhash_dedup"),
            notes=(
                "Disable MinHash/Jaccard near-duplicate filtering while keeping exact "
                "signature tracking and quality-diversity feedback enabled."
            ),
        ),
        ExperimentVariant(
            id="no_disagreement_bd_axis",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="disagreement_bd_axis",
            factors={"disagreement_bd_axis": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "behavioral_descriptor"),
            notes=(
                "Disable backend-disagreement behavioral descriptor axes while keeping "
                "the rest of the descriptor and archive pipeline enabled."
            ),
        ),
        ExperimentVariant(
            id="no_lhs_seeding",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="lhs_seeding",
            factors={"lhs_seeding": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "lhs_seeding"),
            notes=(
                "Disable Latin-hypercube schema initialization while keeping the "
                "closed-loop mutation and scheduler components enabled."
            ),
        ),
        ExperimentVariant(
            id="no_champion_corpus",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="champion_corpus",
            factors={"champion_corpus": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "champion_corpus"),
            notes=(
                "Disable cross-version champion-corpus promotion and grafting while "
                "keeping regular corpus feedback and seed scheduling enabled."
            ),
        ),
        ExperimentVariant(
            id="no_champion_graft_donor",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="champion_graft_donor",
            factors={"champion_graft_donor": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "champion_corpus"),
            notes=(
                "Disable the champion-graft donor bandit while keeping champion-corpus "
                "promotion, regular corpus feedback, and seed scheduling enabled."
            ),
        ),
        ExperimentVariant(
            id="no_backend_pair_learning",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="backend_pair_learning",
            factors={"backend_pair_learning": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "backend_pair_bandit"),
            notes=(
                "Disable backend-pair priority bandit learning while keeping the "
                "adaptive batch scheduler and other feedback scopes enabled."
            ),
        ),
        ExperimentVariant(
            id="no_cost_normalized_reward",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="cost_normalized_reward",
            factors={"cost_normalized_reward": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "cost_normalized_reward"),
            notes=(
                "Disable elapsed-time reward normalization while keeping runtime-cost "
                "penalties and scheduler learning enabled."
            ),
        ),
        ExperimentVariant(
            id="no_active_learning",
            preset="live_deep_organic",
            base_preset="live_deep_organic",
            comparison_role="contrast",
            component_focus="active_learning",
            factors={"active_learning": False},
            oracle_profile="differential",
            rq_tags=("RQ4", "RQ6"),
            analysis_tags=("ablation", "adaptive_component", "active_learning"),
            notes=(
                "Disable uncertainty-driven online exploration while keeping contextual bandit, "
                "runtime-cost learning, and quality-diversity archive enabled."
            ),
        ),
    ),
    comparison_group="adaptive_component_ablation",
    rq_tags=("RQ4", "RQ6"),
    analysis_tags=("ablation", "adaptive_component"),
    counts_as_real_bugs=False,
    notes=(
        "Adaptive-component ablations use support evidence only; candidate bugs require "
        "the same live confirmation pipeline before they can be counted as real bugs."
    ),
    scope_by_target_suite=_suite_scope_map(("datafusion_cross",)),
)


FINAL_COMPARISON_MATRIX = ExperimentMatrix(
    id="baseline_scope_comparison",
    title="Baseline and Scope Comparison",
    track="comparison",
    purpose=(
        "Compare random/guided/metamorphic/workflow presets and SQL/query-engine-only scope "
        "against the cross-ecosystem DataDiffFuzz scope."
    ),
    evidence_mode="comparison",
    target_suites=("embedded_sql", "datafusion_cross", "latest_all_engines", "latest_no_datafusion"),
    variants=(
        ExperimentVariant(
            id="baseline",
            preset="baseline",
            comparison_role="baseline",
            oracle_profile="differential",
            analysis_tags=("comparison", "baseline"),
        ),
        ExperimentVariant(
            id="guided",
            preset="guided",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("enable_guidance",),
            factors={"guidance": "guided"},
            oracle_profile="differential",
            analysis_tags=("comparison", "guided"),
        ),
        ExperimentVariant(
            id="discovery",
            preset="discovery",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("generator_discovery",),
            factors={"generator_profile": "discovery"},
            oracle_profile="differential",
            analysis_tags=("comparison", "generator"),
        ),
        ExperimentVariant(
            id="discovery_guided",
            preset="discovery_guided",
            base_preset="discovery",
            comparison_role="contrast",
            overlays=("enable_guidance", "target_discovery_guided"),
            factors={"generator_profile": "discovery", "guidance": "guided"},
            oracle_profile="differential",
            analysis_tags=("comparison", "generator", "guided"),
        ),
        ExperimentVariant(
            id="metamorphic",
            preset="metamorphic",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("enable_metamorphic_oracle",),
            factors={"metamorphic_oracle": True, "oracle_mode": "both"},
            oracle_profile="both",
            analysis_tags=("comparison", "oracle_complementarity"),
        ),
        ExperimentVariant(
            id="oracle_only_metamorphic",
            preset="oracle_only_metamorphic",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("disable_differential_oracle", "enable_metamorphic_oracle"),
            factors={"differential_oracle": False, "metamorphic_oracle": True, "oracle_mode": "metamorphic"},
            oracle_profile="metamorphic_only",
            analysis_tags=("comparison", "oracle_complementarity"),
        ),
        ExperimentVariant(
            id="workflow",
            preset="workflow",
            base_preset="baseline",
            comparison_role="contrast",
            overlays=("generator_workflow",),
            factors={"generator_profile": "workflow"},
            oracle_profile="differential",
            analysis_tags=("comparison", "workflow"),
        ),
        ExperimentVariant(
            id="workflow_metamorphic",
            preset="workflow_metamorphic",
            base_preset="workflow",
            comparison_role="contrast",
            overlays=("enable_metamorphic_oracle",),
            factors={"generator_profile": "workflow", "metamorphic_oracle": True, "oracle_mode": "both"},
            oracle_profile="both",
            analysis_tags=("comparison", "workflow", "oracle_complementarity"),
        ),
        ExperimentVariant(
            id="live_cross_family",
            preset="live_cross_family",
            comparison_role="contrast",
            scope_kind="cross_ecosystem",
            oracle_profile="differential",
            rq_tags=("RQ1", "RQ6"),
            analysis_tags=("comparison", "cross_ecosystem_scope"),
        ),
    ),
    comparison_group="scope_comparison",
    rq_tags=("RQ1", "RQ4", "RQ6"),
    analysis_tags=("comparison", "scope"),
    counts_as_real_bugs=False,
    notes=(
        "This internal comparison isolates what cross-ecosystem DataFrame/Arrow/SQL coverage "
        "adds beyond SQL/query-engine-oriented testing without introducing an external runner."
    ),
    scope_by_target_suite=_suite_scope_map(("embedded_sql", "datafusion_cross", "latest_all_engines", "latest_no_datafusion")),
)


FINAL_EXPERIMENT_MATRICES: tuple[ExperimentMatrix, ...] = (
    FINAL_VALIDATION_MATRIX,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX,
    FINAL_COMPARISON_MATRIX,
)

FINAL_EXPERIMENT_MATRIX_BY_ID: dict[str, ExperimentMatrix] = {
    matrix.id: matrix
    for matrix in FINAL_EXPERIMENT_MATRICES
}

FINAL_EXPERIMENT_VARIANT_BY_MATRIX_AND_PRESET: dict[tuple[str, str], ExperimentVariant] = {
    (matrix.id, variant.preset): variant
    for matrix in FINAL_EXPERIMENT_MATRICES
    for variant in matrix.variants
}

FINAL_EXPERIMENT_VARIANT_BY_MATRIX_AND_ID: dict[tuple[str, str], ExperimentVariant] = {
    (matrix.id, variant.id): variant
    for matrix in FINAL_EXPERIMENT_MATRICES
    for variant in matrix.variants
}


def final_experiment_matrix(matrix_id: str) -> ExperimentMatrix | None:
    return FINAL_EXPERIMENT_MATRIX_BY_ID.get(str(matrix_id or "").strip())


def registered_experiment_matrix_for_run(
    *,
    evidence_mode: str,
    target_suite: str,
    preset: str,
) -> ExperimentMatrix | None:
    normalized_mode = str(evidence_mode or "").strip()
    normalized_suite = str(target_suite or "").strip()
    normalized_preset = str(preset or "").strip()
    if not normalized_mode or not normalized_suite:
        return None
    matches = [
        matrix
        for matrix in FINAL_EXPERIMENT_MATRICES
        if matrix.evidence_mode == normalized_mode
        and normalized_suite in matrix.target_suites
        and (
            not normalized_preset
            or any(variant.preset == normalized_preset for variant in matrix.variants)
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def registered_experiment_meta_for_run(
    *,
    evidence_mode: str,
    known_bug_id: str = "",
    target_suite: str = "",
    preset: str = "",
    target_version: str = "",
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    normalized_mode = str(evidence_mode or "").strip()
    normalized_suite = str(target_suite or "").strip()
    normalized_preset = str(preset or "").strip()
    if normalized_mode == "historical":
        return resolve_historical_experiment_meta(
            known_bug_id=known_bug_id,
            target_suite=normalized_suite,
            target_version=target_version,
            include_pending=include_pending_historical,
        )
    matrix = registered_experiment_matrix_for_run(
        evidence_mode=normalized_mode,
        target_suite=normalized_suite,
        preset=normalized_preset,
    )
    if matrix is None:
        return {}
    return matrix.to_experiment_meta(target_suites=(normalized_suite,))


def registered_experiment_meta_for_manifest(
    manifest: dict[str, Any] | None,
    *,
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    normalized_mode = str(manifest.get("evidence_mode", "") or "").strip()
    runs = [run for run in manifest.get("runs", []) if isinstance(run, dict)]
    if normalized_mode == "historical":
        first_run = runs[0] if runs else {}
        known_bug_id = str(
            first_run.get("known_bug_id", "") or manifest.get("known_bug_id", "") or ""
        ).strip()
        target_suite = str(
            first_run.get("target_suite", "") or manifest.get("target_suite", "") or ""
        ).strip()
        target_version = str(
            first_run.get("target_version", "") or manifest.get("target_version", "") or ""
        ).strip()
        if not known_bug_id:
            return {}
        return registered_experiment_meta_for_run(
            evidence_mode=normalized_mode,
            known_bug_id=known_bug_id,
            target_suite=target_suite,
            target_version=target_version,
            include_pending_historical=include_pending_historical,
        )
    if not normalized_mode or not runs:
        return {}
    target_suites: list[str] = []
    matrix_ids: set[str] = set()
    for run in runs:
        target_suite = str(run.get("target_suite", "") or manifest.get("target_suite", "") or "").strip()
        preset = str(run.get("preset", "") or "").strip()
        matrix = registered_experiment_matrix_for_run(
            evidence_mode=normalized_mode,
            target_suite=target_suite,
            preset=preset,
        )
        if matrix is None:
            return {}
        if target_suite:
            target_suites.append(target_suite)
        matrix_ids.add(matrix.id)
    if len(matrix_ids) != 1:
        return {}
    matrix = final_experiment_matrix(next(iter(matrix_ids)))
    if matrix is None:
        return {}
    selected_suites = tuple(dict.fromkeys(suite for suite in target_suites if suite))
    return matrix.to_experiment_meta(target_suites=selected_suites or None)


def resolve_final_experiment_variant(
    matrix_id: str,
    *,
    preset: str = "",
    variant_id: str = "",
) -> ExperimentVariant | None:
    normalized_matrix_id = str(matrix_id or "").strip()
    normalized_preset = str(preset or "").strip()
    normalized_variant_id = str(variant_id or "").strip()
    if normalized_matrix_id and normalized_preset:
        variant = FINAL_EXPERIMENT_VARIANT_BY_MATRIX_AND_PRESET.get((normalized_matrix_id, normalized_preset))
        if variant is not None:
            return variant
    if normalized_matrix_id and normalized_variant_id:
        return FINAL_EXPERIMENT_VARIANT_BY_MATRIX_AND_ID.get((normalized_matrix_id, normalized_variant_id))
    return None


def build_final_experiment_variant_config(
    matrix_id: str,
    *,
    preset: str = "",
    variant_id: str = "",
) -> ExperimentConfig:
    variant = resolve_final_experiment_variant(
        matrix_id,
        preset=preset,
        variant_id=variant_id,
    )
    if variant is None:
        raise ValueError(
            f"unknown final experiment variant for matrix={matrix_id!r} preset={preset!r} variant_id={variant_id!r}"
        )
    base_preset = str(variant.base_preset or variant.preset or "").strip()
    overlays = tuple(str(name).strip() for name in variant.overlays if str(name).strip())
    if overlays:
        return build_experiment_config(base_preset, overlays)
    config = build_catalog_preset(base_preset)
    if config is None:
        raise ValueError(f"unknown catalog preset referenced by final experiment variant: {base_preset!r}")
    return config
